"""Device-resident per-Player mission phase state; no physics or command output."""
import math

import torch

from .mission_player import alignment_heading
from .tensor_reference import TensorReference

WARNINGS = (None, 'turn_timeout', 'stop_timeout', 'turn_drift')

PHASES = ('decelerate', 'turn', 'accelerate', 'track', 'settle', 'complete')


def wrap(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


class TensorMissionPhases:
    def __init__(self, parts, count, config, device):
        def tensor(value):
            return torch.tensor(value, dtype=torch.float64, device=device)
        self.duration = tensor([p.duration for p in parts])
        self.end = tensor([p.xy[-1].tolist() for p in parts])
        self.end_yaw = tensor([p.yaw[-1] for p in parts])
        self.start_yaw = tensor([alignment_heading(p, 0.) for p in parts])
        self.config = config
        self.references = [TensorReference(p, [0]*count, device) for p in parts]
        self.stage = torch.zeros(count, dtype=torch.long, device=device)
        self.phase = self.stage.clone()
        self.phase_started = tensor([float('nan')]*count)
        self.stable_since = self.phase_started.clone()
        self.last_act = self.phase_started.clone()
        self.leg_started = tensor([0.]*count)
        self.leg_resume_time = self.leg_started.clone()
        self.anchor = tensor(parts[0].xy[0]).expand(count, 2).clone()
        self.target_yaw = self.start_yaw[0].expand(count).clone()
        self.turn_command = tensor([0.]*count)
        self.resume_heading = self.target_yaw.clone()
        self.warning = self.stage.clone()

    def retime(self, parts, config, now):
        """Install new speed curves while retaining arc progress on every leg."""
        if len(parts) != len(self.references):
            raise ValueError('Retiming cannot change mission topology')
        device, count = self.stage.device, len(self.stage)
        updated = [TensorReference(p, [0]*count, device) for p in parts]
        resume, started, headings = self.leg_resume_time.clone(), self.leg_started.clone(), self.resume_heading.clone()
        moving = (self.phase == 2) | (self.phase == 3)
        for i, (old, new) in enumerate(zip(self.references, updated)):
            selected = self.stage == i
            converted = new.time_at_distance(old.distance_at(self.leg_resume_time))
            resume = torch.where(selected, converted, resume)
            headings = torch.where(selected, new.alignment_heading(converted), headings)
            elapsed = new.time_at_distance(old.distance_at(now-self.leg_started))
            started = torch.where(selected & moving, now-elapsed, started)
        self.leg_resume_time, self.leg_started, self.resume_heading = resume, started, headings
        self.references, self.config = updated, config
        self.duration = torch.tensor([p.duration for p in parts], device=device, dtype=torch.float64)
        self.start_yaw = torch.tensor([alignment_heading(p, 0.) for p in parts], device=device, dtype=torch.float64)

    def recover(self, ids, position, yaw, resume_time, resume_heading, now):
        """Install a recovery sample without canceling the stage or later goals.

        The caller resets FFPD history separately and supplies device-side local
        projection time and alignment heading. Completed tasks stay completed.
        """
        self.anchor[ids] = position
        self.turn_command[ids] = yaw
        self.leg_resume_time[ids] = resume_time
        self.resume_heading[ids] = resume_heading
        self.target_yaw[ids] = resume_heading
        self.last_act[ids] = float('nan')
        active = self.phase[ids] != 5
        self.phase[ids] = torch.where(active, 0, self.phase[ids])
        self.phase_started[ids] = torch.where(active, now, self.phase_started[ids])
        self.stable_since[ids] = torch.where(active, float('nan'), self.stable_since[ids])
        self.warning[ids] = 0

    def _phase(self, mask, phase, now):
        self.phase = torch.where(mask, phase, self.phase)
        self.phase_started = torch.where(mask, now, self.phase_started)
        self.stable_since = torch.where(mask, float('nan'), self.stable_since)

    def _stable(self, active, condition, now):
        self.stable_since = torch.where(active & ~condition, float('nan'), self.stable_since)
        self.stable_since = torch.where(active & condition & self.stable_since.isnan(), now, self.stable_since)
        return active & condition & (now-self.stable_since >= .3-1e-8)

    def step(self, position, velocity, actual_yaw, previous_command, now, transition_ready):
        first = self.phase_started.isnan()
        self.phase_started = torch.where(first, now, self.phase_started)
        self.anchor = torch.where(first[:, None], position, self.anchor)
        self.turn_command = torch.where(first, actual_yaw, self.turn_command)
        dt = torch.where(self.last_act.isnan(), self.config.dt, (now-self.last_act).clamp_min(0.))
        self.last_act = torch.full_like(self.last_act, now)
        # Recovery installs the resumed leg's alignment heading separately.
        self.target_yaw = torch.where((self.phase == 0) | (self.phase == 1),
                                      self.resume_heading, self.target_yaw)
        moving = (self.phase == 2) | (self.phase == 3)
        elapsed = now-self.leg_started
        ending = moving & (elapsed >= self.duration[self.stage])
        self.anchor = torch.where(ending[:, None], self.end[self.stage], self.anchor)
        self.target_yaw = torch.where(ending, self.end_yaw[self.stage], self.target_yaw)
        self.turn_command = torch.where(ending, actual_yaw, self.turn_command)
        self.phase = torch.where(moving & ~ending & (elapsed >= 1.), 3, self.phase)
        self._phase(ending, 4, now)
        distance = torch.linalg.vector_norm(position-self.anchor, dim=-1)
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        stopped = ((speed < .12) & (torch.linalg.vector_norm(previous_command, dim=-1)
                    <= self.config.max_acceleration*self.config.dt) & (distance < .3))
        stop_phase = (self.phase == 0) | (self.phase == 4)
        stable = self._stable(stop_phase, stopped & ((self.phase != 4) | transition_ready), now)
        complete = stable & (self.phase == 4) & (self.stage == len(self.duration)-1)
        advance = stable & (self.phase == 4) & ~complete
        self.stage = self.stage+advance.long()
        self.leg_resume_time = torch.where(advance, 0., self.leg_resume_time)
        self.resume_heading = torch.where(advance, self.start_yaw[self.stage], self.resume_heading)
        self.target_yaw = torch.where(advance, self.resume_heading, self.target_yaw)
        self._phase(complete, 5, now)
        self._phase(stable & ~complete, 1, now)
        self.turn_command = torch.where(stable & ~complete, actual_yaw, self.turn_command)
        turning = self.phase == 1
        heading_error = wrap(self.target_yaw-actual_yaw)
        correction = torch.where(heading_error.abs() < math.radians(30),
                                 (.5*heading_error).clamp(-math.radians(10), math.radians(10)), 0.)
        remaining = wrap(self.target_yaw+correction-self.turn_command)
        bound = math.radians(30)*dt
        proposed = self.turn_command+torch.minimum(torch.maximum(remaining, -bound), bound)
        facing = actual_yaw+wrap(proposed-actual_yaw).clamp(-math.radians(30), math.radians(30))
        self.turn_command = torch.where(turning, facing, self.turn_command)
        aligned = heading_error.abs() < math.radians(10)
        accelerate = self._stable(turning, aligned & (speed < .15) & transition_ready, now)
        self.leg_started = torch.where(accelerate, now-self.leg_resume_time, self.leg_started)
        self._phase(accelerate, 2, now)
        # Preserve the original alert precedence, without any pause/hold action.
        self.warning = torch.where((self.phase == 1) & (now-self.phase_started > 20.), 1, 0)
        self.warning = torch.where(((self.phase == 0) | (self.phase == 4))
                                   & (now-self.phase_started > 20.), 2, self.warning)
        self.warning = torch.where((self.phase == 1) & (distance > .6), 3, self.warning)
