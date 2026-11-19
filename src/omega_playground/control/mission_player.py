"""Ordered tasks: stop, align using measured heading, then track the next leg."""
import math
from dataclasses import replace

import numpy as np
import torch

from omega_playground.control.ffpd import FFPD, FFPDConfig
from omega_playground.control.locomotion import MODES, validate_locomotion
from omega_playground.control.reference import ReferenceTrajectory
from omega_playground.control.route_player import RoutePlayer
from omega_playground.policy.base import JoystickBatch
from omega_playground.policy.schedule import Schedule


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def alignment_heading(part, resume_time, lookahead_m=.25):
    """Align to local route progress, not a centimetre-scale snap connector.

    Geometry and execution timing stay unchanged. Each reversal is already a
    separate part, so the lookahead never crosses an explicit turning boundary.
    """
    t = float(np.clip(resume_time, 0., part.duration))
    i = min(int(np.searchsorted(part.times, t, side='right'))-1, len(part.times)-2)
    tau = t-part.times[i]
    acceleration = (part.speed[i+1]-part.speed[i])/(part.times[i+1]-part.times[i])
    distance = part.s[i]+part.speed[i]*tau+.5*acceleration*tau*tau
    start, fallback = part.sample(t)
    ahead = min(distance+lookahead_m, part.length)
    target = np.array([np.interp(ahead, part.s, part.xy[:, axis]) for axis in range(2)])
    delta = target-start
    return float(math.atan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-6 else fallback


def split_sharp_turns(route, threshold=120.):
    """Keep 180-degree topology; split it into independently executable legs."""
    points = np.asarray(route, float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Route must contain finite XY points')
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]]
    if len(points) < 2:
        return []
    pieces, begin = [], 0
    for i in range(1, len(points) - 1):
        u, v = points[i] - points[i-1], points[i+1] - points[i]
        cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        if cosine <= math.cos(math.radians(threshold)):
            pieces.append(points[begin:i+1].tolist())
            begin = i
    pieces.append(points[begin:].tolist())
    return pieces


class MissionReference:
    """Static geometry for validation/display; execution clocks belong to each leg."""
    def __init__(self, parts):
        self.parts = parts
        self.ends = np.cumsum([p.duration for p in parts])
        self.starts = np.r_[0., self.ends[:-1]]
        self.duration = float(self.ends[-1])
        self.xy = np.concatenate([p.xy for p in parts])
        self.times = np.concatenate([p.times + t for p, t in zip(parts, self.starts)])
        self.offsets = np.array([[0., 0.]])
        self.closed = False
        self.nominal_speed, self.corner_radius = parts[0].nominal_speed, parts[0].corner_radius

    def sample(self, t, role=0):
        index = min(int(np.searchsorted(self.ends, t, side='right')), len(self.parts)-1)
        return self.parts[index].sample(t-self.starts[index], role)

    def describe(self):
        return {'schema': 'omega.reference/1', 'closed': False, 'legs': len(self.parts),
                'duration_s': self.duration, 'duration_excludes_turns': True,
                'length_m': sum(p.length for p in self.parts), 'slot_offsets': self.offsets.tolist()}


class MissionPlayer(RoutePlayer):
    """Single-robot FFPD task with explicit braking and measured-heading turn gates."""
    def __init__(self, spec, device):
        self.name = 'mission_ffpd'
        self.lateral_control = spec.get('lateral_control', True)
        self.num_agents, self.device = 1, device
        self.schedule = Schedule(trigger='periodic', period_s=1.0)
        self.config = FFPDConfig(**{**spec.get('ffpd', {}), 'dt': 1.0})
        if abs(self.schedule.interval_s() - self.config.dt) > 1e-9:
            raise ValueError('FFPD dt must match the Player schedule')
        options = {**spec.get('reference', {}), 'closed': False, 'slot_offsets': [[0., 0.]],
                   'max_speed': self.config.max_speed, 'max_acceleration': self.config.max_acceleration}
        parts, labels = [], []
        for index, leg in enumerate(spec.get('legs', [spec['route']])):
            for piece in split_sharp_turns(leg):
                parts.append(ReferenceTrajectory(piece, **options))
                labels.append('approach' if index < spec.get('approach_legs', 0) else 'route')
        if not parts:
            raise ValueError('Task has no traversable legs')
        self.reference = MissionReference(parts)
        self.labels = labels
        self.controller = FFPD(1, self.config)
        self.start_times = np.zeros(1)
        self.initialized = False
        self.hold, self.navigation, self.warning = None, None, None
        self.tracking_divergence_hold = True
        self.safety_holds = True
        self.transition_ready = True  # Formation may wait for members at a leg boundary.
        self.warnings = []
        self.motion_warning = None
        self.idle = np.zeros(1, dtype=bool)
        self.idle_xy = np.zeros((1, 2))
        self.idle_yaw = np.zeros(1)
        self.projections = {}
        self.clearance = float(spec.get('clearance_m', .6))
        self.divergence, self.divergence_seconds = 2., 1.
        self.diverged_since = np.full(1, np.nan)
        self.task_mode, self.patrol_id = spec.get('task_mode', 'navigate'), spec.get('patrol_id')
        self.sonic_mode = spec.get("sonic_mode", "walk")
        self._restart()

    def _restart(self):
        self.stage = 0
        self.phase = 'decelerate'
        self.phase_started = None
        self.leg_started = 0.
        self.leg_resume_time = 0.
        self.stable_since = None
        self.anchor = self.reference.parts[0].xy[0].copy()
        self.turn_command = None
        self.target_yaw = alignment_heading(self.reference.parts[0], 0.)
        self.last_act = None

    def reset(self, agent_ids, view):
        super().reset(agent_ids, view)
        self._restart()

    def recover_task(self, agent_ids, view, sim_time):
        part = self.reference.parts[self.stage]
        resume_time = self.recovery_time(0, view.root_pos[0, :2].cpu().numpy(), part)
        projections = self.projections.copy()
        RoutePlayer.reset(self, torch.as_tensor(agent_ids, device=self.device), view)
        self.projections = projections
        self.controller.previous[:] = 0.
        self.initialized = True
        self.hold = self.motion_warning = None
        self.anchor = view.root_pos[0, :2].cpu().numpy().copy()
        x, y, z, w = getattr(view, "heading_rot", view.root_rot)[0].cpu().numpy()
        self.turn_command = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        self.leg_resume_time = resume_time
        self.target_yaw = alignment_heading(part, resume_time)
        self.last_act = None
        self.start_times[:] = sim_time
        # Completed/idle tasks stay finished. Active tasks retain stage and all later goals.
        if self.phase != 'complete' and not self.idle.all():
            self._phase('decelerate', sim_time)

    def status(self):
        return {'mode': 'idle' if self.idle.all() else self.task_mode, 'patrol_id': self.patrol_id, 'phase': 'idle' if self.idle.all() else self.phase,
                'segment': self.stage + 1, 'segments': len(self.reference.parts),
                'section': self.labels[self.stage],
                'target': self.reference.parts[self.stage].xy[-1].tolist()}

    def configure_locomotion(self, mode, speed, sim_time):
        speed = validate_locomotion(mode, speed)
        config = replace(self.config, max_speed=max(1., speed+.35))
        parts = [p.retimed(speed, config.max_speed, config.max_acceleration) for p in self.reference.parts]
        resume_distance = self.reference.parts[self.stage].distance_at(self.leg_resume_time)
        self.leg_resume_time = parts[self.stage].time_at_distance(resume_distance)
        if self.phase in ('track', 'accelerate'):
            distance = self.reference.parts[self.stage].distance_at(sim_time-self.leg_started)
            self.leg_started = sim_time-parts[self.stage].time_at_distance(distance)
        self.reference = MissionReference(parts)
        self.config = self.controller.config = config
        self.sonic_mode = mode

    def targets(self, sim_time):
        if self.idle.all():
            return [(self.idle_xy[0].copy(), self.idle_yaw[0])]
        if self.phase in ('track', 'accelerate'):
            return [self.reference.parts[self.stage].sample(sim_time - self.leg_started)]
        return [(self.anchor.copy(), float(self.target_yaw))]

    def _phase(self, phase, now):
        self.phase, self.phase_started, self.stable_since = phase, now, None

    def _stable(self, condition, now, duration=.3):
        if not condition:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = now
        return now - self.stable_since >= duration - 1e-8

    def alert(self, reason):
        if self.safety_holds:
            self.hold = reason
        else:
            self.motion_warning = reason

    def act(self, view, task, sim_time):
        self.motion_warning = None
        if self.idle.all():
            return super().act(view, task, sim_time)
        p = view.root_pos[0, :2].detach().cpu().numpy()
        v = view.root_vel[0, :2].detach().cpu().numpy()
        x, y, z, w = getattr(view, "heading_rot", view.root_rot)[0].detach().cpu().numpy()
        actual_yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        now = sim_time
        if self.phase_started is None:
            self.phase_started = now
            self.anchor = p.copy()
            self.turn_command = actual_yaw
            if not self.initialized:
                self.controller.reset([0], [v])
                self.initialized = True
        dt = self.config.dt if self.last_act is None else max(0., now-self.last_act)
        self.last_act = now
        part = self.reference.parts[self.stage]
        self.target_yaw = alignment_heading(part, self.leg_resume_time) if self.phase in ('decelerate', 'turn') else self.target_yaw
        if self.phase in ('track', 'accelerate'):
            if now - self.leg_started >= part.duration:
                self.anchor = part.xy[-1].copy()
                self.target_yaw = float(part.yaw[-1])
                self.turn_command = actual_yaw
                self._phase('settle', now)
            elif now - self.leg_started >= 1.:
                self.phase = 'track'
        distance = float(np.linalg.norm(p-self.anchor))
        if self.phase in ('decelerate', 'settle'):
            stopped = (np.linalg.norm(v) < .12 and np.linalg.norm(self.controller.previous) <= self.config.max_acceleration * self.config.dt
                       and distance < .3)
            if self._stable(stopped and (self.phase != 'settle' or self.transition_ready), now):
                if self.phase == 'settle':
                    if self.stage == len(self.reference.parts)-1:
                        self._phase('complete', now)
                    else:
                        self.stage += 1
                        self.leg_resume_time = 0.
                        part = self.reference.parts[self.stage]
                        self.target_yaw = alignment_heading(part, 0.)
                        self._phase('turn', now)
                else:
                    self._phase('turn', now)
                if self.phase != 'complete':
                    self.turn_command = actual_yaw
        if self.phase == 'turn':
            # Never ask SONIC for a sudden large facing change. Each command is
            # <=30 degrees ahead of measured heading and slews <=30 degrees/s.
            # Learned idle turning can retain a small steady heading bias.
            # Keep the arrival tolerance, but feed measured error back into
            # the facing request; existing slew/ahead bounds still apply.
            heading_error = wrap(self.target_yaw-actual_yaw)
            correction = (np.clip(.5*heading_error, -math.radians(10), math.radians(10))
                          if abs(heading_error) < math.radians(30) else 0.)
            remaining = wrap(self.target_yaw+correction-self.turn_command)
            proposed = self.turn_command + np.clip(remaining, -math.radians(30)*dt, math.radians(30)*dt)
            self.turn_command = actual_yaw + np.clip(wrap(proposed-actual_yaw),
                                                      -math.radians(30), math.radians(30))
            aligned = abs(wrap(self.target_yaw-actual_yaw)) < math.radians(10)
            if self._stable(aligned and np.linalg.norm(v) < .15 and self.transition_ready, now):
                self.leg_started = now-self.leg_resume_time
                self._phase('accelerate', now)
            elif now-self.phase_started > 20.:
                self.alert({'code': 'turn_timeout', 'agents': [0], 'sim_time': now})
        if self.phase in ('decelerate', 'settle') and now-self.phase_started > 20.:
            self.alert({'code': 'stop_timeout', 'agents': [0], 'sim_time': now})
        if self.phase == 'turn' and distance > .6:
            self.alert({'code': 'turn_drift', 'agents': [0], 'sim_time': now})
        measured = self.measure(view, now)
        reference_xy = measured['reference_xy']
        if self.phase in ('decelerate', 'settle') and distance < .3:
            # Inside arrival tolerance, brake velocity instead of fighting the
            # residual position error with an endless small walking command.
            reference_xy = np.array([p])
        command = self.controller.update(reference_xy, measured['reference_velocity_xy'],
                                         view.root_pos.detach().cpu().numpy(), view.root_vel.detach().cpu().numpy(),
                                         **(self.lateral_feedback(measured) if self.phase in ('track', 'accelerate') else {}))
        speed = float(np.linalg.norm(command[0]))
        movement = np.r_[command[0]/max(speed, 1e-8), 0.]
        facing = self.targets(now)[0][1]
        if self.lateral_control and self.phase in ('track', 'accelerate'):
            facing = math.atan2(*measured['route_tangent'][0][::-1])
        mode = MODES[self.sonic_mode]["id"]
        # Run cannot reliably produce the small/backward corrections needed to
        # dock at a node. Use Walk for the final braking ramp and settling,
        # while preserving the reference clock and acceleration limiter.
        approaching = (self.phase in ('track', 'accelerate')
                       and part.duration - (now-self.leg_started) <= 3.)
        if self.sonic_mode == 'run' and (approaching or self.phase in ('decelerate', 'settle')):
            mode = MODES['walk']['id']
        if self.phase in ('decelerate', 'settle'):
            facing = self.turn_command
            if distance < .3 and speed <= self.config.max_acceleration * self.config.dt:
                mode, speed, movement = 0, -1., np.zeros(3)
        elif self.phase == 'turn':
            # SONIC's idle mode honors facing without forward walking intent.
            mode, speed, movement, facing = 0, -1., np.zeros(3), self.turn_command
            self.controller.previous[:] = 0.
        elif self.phase == 'complete':
            # Completion holds the settled facing; do not rotate the formation
            # back to the nominal route tangent after announcing arrival.
            mode, speed, movement, facing = 0, -1., np.zeros(3), self.turn_command
        elif now < self.start_times[0]:
            mode, speed, movement = 0, -1., np.zeros(3)
        return JoystickBatch(mode=torch.tensor([mode], device=self.device),
                             speed=torch.tensor([speed], device=self.device, dtype=torch.float64),
                             movement=torch.tensor([movement.tolist()], device=self.device, dtype=torch.float64),
                             facing=torch.tensor([[math.cos(facing), math.sin(facing), 0.]], device=self.device, dtype=torch.float64),
                             vertical_speed=torch.zeros(1, device=self.device),
                             height=torch.full((1,), -1., device=self.device))
