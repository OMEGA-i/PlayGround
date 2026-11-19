"""GPU route sampling -> local projection -> FFPD, below task orchestration."""
from .tensor_ffpd import TensorFFPD
from .tensor_reference import TensorReference
from .tensor_route_deviation import TensorRouteProjection


class TensorRouteFeedback:
    def __init__(self, reference, roles, config, device):
        self.reference = TensorReference(reference, roles, device)
        self.projection = TensorRouteProjection(reference, roles, device)
        self.controller = TensorFFPD(len(roles), config, device)

    def update(self, position, velocity, elapsed):
        target, yaw = self.reference.sample(elapsed)
        ahead, _ = self.reference.sample(elapsed+self.controller.config.dt)
        target_velocity = (ahead-target)/self.controller.config.dt
        error, progress = self.projection.project(position[..., :2])
        command = self.controller.update(target, target_velocity, position, velocity,
            route_error=error, route_tangent=self.projection.tangent,
            route_progress_error=self.reference.distance_at(elapsed)-progress)
        return command, self.projection.tangent, yaw

    def reset(self, ids, velocity):
        self.controller.reset(ids, velocity)
        self.projection.reset(ids)
