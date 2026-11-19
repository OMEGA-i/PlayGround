"""Single-robot Web lifecycle adapter around the GPU mission command core.

Host route validation/status/warnings stay explicit diagnostic boundaries.
Only act() supplies commands; it never invokes the NumPy Player controller.
"""
import numpy as np
import torch

from .mission_player import MissionPlayer
from .tensor_mission_player import TensorMissionPlayer
from .tensor_mission_phases import PHASES, WARNINGS
from omega_playground.policy.base import JoystickBatch


class TensorWebPlayer(MissionPlayer):
    def __init__(self, spec, device):
        super().__init__(spec, device)
        self.core = TensorMissionPlayer(spec, device)
        self._seeded = False
        self._idle_command = None
        self._diagnostics_dirty = True

    def _sync(self):
        """Read diagnostic state; never used inside act()."""
        if not self._diagnostics_dirty:
            return
        s = self.core.state
        self.stage, phase, warning = torch.stack((s.stage[0], s.phase[0], s.warning[0])).cpu().tolist()
        self.phase = PHASES[phase]
        scalars = torch.stack((s.phase_started[0], s.leg_started[0], s.leg_resume_time[0],
                               s.stable_since[0], s.turn_command[0], s.target_yaw[0], s.last_act[0])).cpu().tolist()
        (self.phase_started, self.leg_started, self.leg_resume_time, self.stable_since,
         self.turn_command, self.target_yaw, self.last_act) = [None if np.isnan(v) else v for v in scalars]
        self.anchor = s.anchor[0].cpu().numpy().copy()
        if self._seeded:
            self.controller.previous[:] = self.core.controller.previous.cpu().numpy()
            self.controller.report = {k:v.cpu().numpy() for k,v in self.core.controller.report.items()}
        self.motion_warning = ({'code': WARNINGS[warning], 'agents': [0]} if warning else None)
        self._diagnostics_dirty = False

    def act(self, view, task, sim_time):
        if self.idle.all():
            return self._idle_command
        if not self._seeded:
            if self.initialized:
                self.core.controller.previous.copy_(torch.as_tensor(self.controller.previous, device=self.device))
                self.core.command_initialized[:] = True
            self._seeded = True
            self.initialized = True
        self.core.transition_ready[:] = self.transition_ready
        self._diagnostics_dirty = True
        return self.core.act(view, task, sim_time)

    def status(self):
        self._sync()
        return super().status()

    def measure(self, view, sim_time):
        self._sync()
        return super().measure(view, sim_time)

    def reset(self, agent_ids, view):
        self._diagnostics_dirty = True
        self.core.reset(agent_ids, view)
        self.projections.clear()
        self.diverged_since[:] = np.nan
        self.hold = self.motion_warning = None
        self._seeded = self.initialized = True
        self._sync()

    def start_aligned(self, view, sim_time=0.):
        """Start a freshly reset, stationary task already aligned to its first leg."""
        s = self.core.state
        rotation = getattr(view, 'heading_rot', view.root_rot).double()
        x, y, z, w = rotation.unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        error = torch.atan2(torch.sin(yaw-s.start_yaw[0]), torch.cos(yaw-s.start_yaw[0]))
        speed = torch.linalg.vector_norm(view.root_vel[..., :2], dim=-1)
        distance = torch.linalg.vector_norm(view.root_pos[..., :2]-s.anchor, dim=-1)
        if (not bool((s.stage == 0).all()) or not bool((s.phase == 0).all())
                or not bool((error.abs() < np.deg2rad(10)).all())
                or not bool((speed < .15).all()) or not bool((distance < .1).all())):
            raise ValueError('Aligned startup requires a fresh stationary state at the first route heading')
        s.phase[:] = 2
        s.phase_started[:] = sim_time
        s.leg_started[:] = sim_time
        s.turn_command[:] = yaw
        self._diagnostics_dirty = True

    def recover_task(self, agent_ids, view, sim_time):
        self._diagnostics_dirty = True
        self.core.recover_task(agent_ids, view, sim_time)
        self.start_times[:] = sim_time
        self.hold = None
        self._seeded = self.initialized = True
        self._sync()

    def configure_locomotion(self, mode, speed, sim_time):
        self._diagnostics_dirty = True
        self.core.configure_locomotion(mode, speed, sim_time)
        self.reference = self.core.reference
        self.config = self.controller.config = self.core.config
        self.sonic_mode = mode
        self._sync()

    def recover_idle(self, agent_ids, view):
        self._diagnostics_dirty = True
        super().recover_idle(agent_ids, view)
        self.core.controller.previous[:] = 0.
        yaw = torch.as_tensor(self.idle_yaw, device=self.device)
        zero = torch.zeros_like(yaw)
        self._idle_command = JoystickBatch(torch.zeros(1, dtype=torch.long, device=self.device),
            torch.full_like(yaw, -1.), torch.zeros(1,3,device=self.device),
            torch.stack((yaw.cos(),yaw.sin(),zero),-1),zero,torch.full_like(yaw,-1.))
