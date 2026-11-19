"""Device command path for startup routes; host lifecycle/diagnostics are explicit."""
import torch

from .route_player import RoutePlayer
from .tensor_route_feedback import TensorRouteFeedback
from .locomotion import MODES
from omega_playground.policy.base import JoystickBatch


class TensorRoutePlayer(RoutePlayer):
    def __init__(self, spec, device):
        super().__init__(spec, 1, device)
        self.feedback = TensorRouteFeedback(self.reference, [0], self.config, device)
        self._seeded = False
        self._upload_lifecycle()

    def _upload_lifecycle(self):
        self._diagnostics_dirty = True
        self.device_starts = torch.as_tensor(self.start_times.copy(), device=self.device)
        self.device_idle = torch.as_tensor(self.idle.copy(), device=self.device)
        self.device_idle_yaw = torch.as_tensor(self.idle_yaw.copy(), device=self.device)

    def _sync(self):
        if not self._diagnostics_dirty:
            return
        if self._seeded:
            self.controller.previous[:] = self.feedback.controller.previous.cpu().numpy()
            self.controller.report = {k:v.cpu().numpy() for k,v in self.feedback.controller.report.items()}
        self._diagnostics_dirty = False

    def measure(self, view, sim_time):
        self._sync()
        return super().measure(view, sim_time)

    def reset(self, agent_ids, view):
        super().reset(agent_ids, view)
        ids = torch.as_tensor(agent_ids, device=self.device)
        self.feedback.reset(ids, view.root_vel[ids])
        self._upload_lifecycle()

    def recover_task(self, agent_ids, view, sim_time):
        super().recover_task(agent_ids, view, sim_time)
        self.feedback.controller.previous.copy_(torch.as_tensor(self.controller.previous, device=self.device))
        # Recovery preserves the local arc cursor; resetting it to the start
        # would steer a mid-route robot toward an earlier crossing.
        projection = self.projections[(id(self.reference.xy), 0)]
        for name in ('progress', 'reference_distance', 'tangent'):
            getattr(self.feedback.projection, name)[0] = torch.as_tensor(
                getattr(projection, name), device=self.device)
        self._seeded = True
        self._upload_lifecycle()

    def recover_idle(self, agent_ids, view):
        super().recover_idle(agent_ids, view)
        self.feedback.controller.previous[agent_ids] = 0.
        self._upload_lifecycle()

    def configure_locomotion(self, mode, speed, sim_time):
        self._sync()
        super().configure_locomotion(mode, speed, sim_time)
        self.feedback = TensorRouteFeedback(self.reference, [0], self.config, self.device)
        self.feedback.controller.previous.copy_(torch.as_tensor(self.controller.previous, device=self.device))
        self._upload_lifecycle()

    def act(self, view, task, sim_time):
        self._diagnostics_dirty = True
        if not self._seeded:
            if self.initialized:
                self.feedback.controller.previous.copy_(torch.as_tensor(self.controller.previous, device=self.device))
            else:
                self.feedback.controller.reset(torch.arange(1, device=self.device), view.root_vel)
            self._seeded = self.initialized = True
        elapsed = sim_time-self.device_starts
        f = self.feedback
        if self.lateral_control:
            command, tangent, yaw = f.update(view.root_pos, view.root_vel, elapsed)
        else:
            target, yaw = f.reference.sample(elapsed)
            ahead, _ = f.reference.sample(elapsed+self.config.dt)
            command = f.controller.update(target, (ahead-target)/self.config.dt, view.root_pos, view.root_vel)
            tangent = torch.stack((yaw.cos(), yaw.sin()), -1)
        speed = torch.linalg.vector_norm(command, dim=-1)
        zero = torch.zeros_like(speed)
        movement = torch.cat((command/speed[:,None].clamp_min(1e-8), zero[:,None]), -1)
        facing = torch.cat((tangent, zero[:,None]), -1)
        waiting = (elapsed < 0) | (elapsed >= self.reference.duration) | self.device_idle
        movement = torch.where(waiting[:,None], 0., movement)
        speed = torch.where(waiting, -1., speed)
        idle_facing = torch.stack((self.device_idle_yaw.cos(), self.device_idle_yaw.sin(), zero), -1)
        facing = torch.where(self.device_idle[:,None], idle_facing, facing)
        f.controller.previous = torch.where(self.device_idle[:,None], 0., f.controller.previous)
        mode = torch.where(waiting, 0, MODES[self.sonic_mode]['id']).long()
        return JoystickBatch(mode, speed, movement, facing, zero, torch.full_like(zero, -1.))
