"""Device-resident mission command core; Web orchestration adapter is separate."""
from dataclasses import replace

import torch

from omega_playground.policy.base import JoystickBatch
from omega_playground.policy.schedule import Schedule
from .ffpd import FFPDConfig
from .locomotion import MODES, validate_locomotion
from .mission_player import MissionPlayer, MissionReference
from .tensor_ffpd import TensorFFPD, limit
from .tensor_mission_phases import TensorMissionPhases
from .tensor_route_deviation import TensorRouteProjection


class TensorMissionPlayer:
    name = 'tensor_mission_ffpd'
    schedule = Schedule(trigger='periodic', period_s=1.)

    def __init__(self, spec, device, count=1):
        # Geometry preparation is a task-install operation. No CPU Player runs.
        prepared = MissionPlayer(spec, torch.device('cpu'))
        self.parts = prepared.reference.parts
        self.reference = prepared.reference
        self.labels = prepared.labels
        self.task_mode, self.patrol_id = prepared.task_mode, prepared.patrol_id
        self.config = FFPDConfig(**{**spec.get('ffpd', {}), 'dt': 1.})
        self.num_agents, self.device = count, torch.device(device)
        self.lateral_control = spec.get('lateral_control', True)
        self.mode = MODES[spec.get('sonic_mode', 'walk')]['id']
        self.state = TensorMissionPhases(self.parts, count, self.config, self.device)
        self.controller = TensorFFPD(count, self.config, self.device)
        self.projections = [TensorRouteProjection(p, [0]*count, self.device) for p in self.parts]
        self.transition_ready = torch.ones(count, dtype=torch.bool, device=self.device)
        self.command_initialized = torch.zeros(count, dtype=torch.bool, device=self.device)

    def reset(self, agent_ids, view):
        ids = torch.as_tensor(agent_ids, device=self.device, dtype=torch.long)
        fresh = TensorMissionPhases(self.parts, self.num_agents, self.config, self.device)
        for name in ('stage', 'phase', 'phase_started', 'stable_since', 'last_act',
                     'leg_started', 'leg_resume_time', 'anchor', 'target_yaw',
                     'turn_command', 'resume_heading', 'warning'):
            getattr(self.state, name)[ids] = getattr(fresh, name)[ids]
        self.controller.reset(ids, view.root_vel[ids])
        self.command_initialized[ids] = True
        for projection in self.projections:
            projection.reset(ids)

    def recover_task(self, agent_ids, view, sim_time):
        ids = torch.as_tensor(agent_ids, device=self.device, dtype=torch.long)
        selected = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        selected[ids] = True
        position = view.root_pos[..., :2].double()
        x,y,z,w = getattr(view, 'heading_rot', view.root_rot).double().unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        resume, heading = self.state.leg_resume_time.clone(), self.state.resume_heading.clone()
        for i, (reference, projection) in enumerate(zip(self.state.references, self.projections)):
            active = selected & (self.state.stage == i)
            _, distance = projection.project(position, active=active)
            elapsed = reference.time_at_distance(distance)
            resume = torch.where(active, elapsed, resume)
            heading = torch.where(active, reference.alignment_heading(elapsed), heading)
        self.state.recover(ids, position[ids], yaw[ids], resume[ids], heading[ids], sim_time)
        self.controller.previous[ids] = 0.
        self.command_initialized[ids] = True

    def configure_locomotion(self, mode, speed, sim_time):
        speed = validate_locomotion(mode, speed)
        config = replace(self.config, max_speed=max(1., speed+.35))
        parts = [p.retimed(speed, config.max_speed, config.max_acceleration) for p in self.parts]
        self.state.retime(parts, config, sim_time)
        self.parts, self.reference = parts, MissionReference(parts)
        self.config = self.controller.config = config
        self.mode = MODES[mode]['id']

    def act(self, view, task, sim_time):
        position, velocity = view.root_pos[..., :2].double(), view.root_vel[..., :2].double()
        rotation = getattr(view, 'heading_rot', view.root_rot).double()
        x,y,z,w = rotation.unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        first = ~self.command_initialized
        self.controller.previous = torch.where(first[:, None], limit(velocity, self.config.max_speed),
                                               self.controller.previous)
        self.command_initialized[:] = True
        s = self.state
        s.step(position, velocity, yaw, self.controller.previous, sim_time, self.transition_ready)
        moving = (s.phase == 2) | (s.phase == 3)
        stopping = (s.phase == 0) | (s.phase == 4)
        close = torch.linalg.vector_norm(position-s.anchor, dim=-1) < .3
        target, ahead, facing = s.anchor.clone(), s.anchor.clone(), s.target_yaw.clone()
        route_error, tangent = torch.zeros_like(position), torch.zeros_like(position)
        progress_error = torch.zeros_like(yaw)
        elapsed = sim_time-s.leg_started
        for i, (reference, projection) in enumerate(zip(s.references, self.projections)):
            selected = s.stage == i
            xy, angle = reference.sample(elapsed)
            next_xy, _ = reference.sample(elapsed+self.config.dt)
            tracking = selected & moving
            target = torch.where(tracking[:, None], xy, target)
            ahead = torch.where(tracking[:, None], next_xy, ahead)
            facing = torch.where(tracking, angle, facing)
            error, distance = projection.project(position, active=selected)
            route_error = torch.where(selected[:, None], error, route_error)
            tangent = torch.where(selected[:, None], projection.tangent, tangent)
            progress_error = torch.where(selected, reference.distance_at(elapsed)-distance, progress_error)
        reference_velocity = (ahead-target)/self.config.dt
        target = torch.where((stopping & close)[:, None], position, target)
        command = self.controller.update(target, reference_velocity, position, velocity,
            route_error=route_error, route_tangent=tangent, route_progress_error=progress_error,
            route_mask=moving & self.lateral_control)
        speed = torch.linalg.vector_norm(command, dim=-1)
        movement = torch.cat((command/speed.clamp_min(1e-8)[:, None], torch.zeros_like(speed[:, None])), dim=-1)
        if self.lateral_control:
            facing = torch.where(moving, torch.atan2(tangent[:, 1], tangent[:, 0]), facing)
        facing = torch.where(~moving, s.turn_command, facing)
        mode = torch.full_like(s.stage, self.mode)
        if self.mode == MODES['run']['id']:
            approaching = moving & (s.duration[s.stage]-elapsed <= 3.)
            mode = torch.where(approaching | stopping, MODES['walk']['id'], mode)
        idle = ((stopping & close & (speed <= self.config.max_acceleration*self.config.dt))
                | (s.phase == 1) | (s.phase == 5))
        self.controller.previous = torch.where((s.phase == 1)[:, None], 0., self.controller.previous)
        return JoystickBatch(mode=torch.where(idle, 0, mode), speed=torch.where(idle, -1., speed),
            movement=torch.where(idle[:, None], 0., movement),
            facing=torch.stack((facing.cos(), facing.sin(), torch.zeros_like(facing)), dim=-1),
            vertical_speed=torch.zeros_like(speed), height=torch.full_like(speed, -1.))

    def status(self):
        """Explicit diagnostic readback; never called by the command kernel."""
        if self.num_agents != 1:
            raise ValueError('Use per-Player status for batched tasks')
        from .tensor_mission_phases import PHASES, WARNINGS
        stage, phase, warning = torch.stack((self.state.stage[0], self.state.phase[0],
                                            self.state.warning[0])).cpu().tolist()
        return {'mode': self.task_mode, 'patrol_id': self.patrol_id,
                'phase': PHASES[phase], 'segment': stage+1, 'segments': len(self.parts),
                'section': self.labels[stage], 'target': self.parts[stage].xy[-1].tolist(),
                'warning': WARNINGS[warning]}
