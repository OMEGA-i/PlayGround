"""Device-resident, batched FFPD feedback; task transitions remain caller-owned."""
import torch


def limit(value, maximum):
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-12)
    return value * (maximum / norm).clamp_max(1.)


class TensorFFPD:
    def __init__(self, count, config, device):
        self.config = config
        self.previous = torch.zeros(count, 2, dtype=torch.float64, device=device)
        self.report = {}

    def _xy(self, value):
        if not isinstance(value, torch.Tensor) or value.device != self.previous.device:
            raise ValueError('FFPD states must already be tensors on the controller device')
        value = value[..., :2].to(dtype=torch.float64)
        if value.shape != self.previous.shape:
            raise ValueError('FFPD expects per-agent XY states')
        torch._assert_async(torch.isfinite(value).all(), 'FFPD expects finite states')
        return value

    def reset(self, ids, velocity):
        if velocity.device != self.previous.device:
            raise ValueError('Reset velocity must be on the controller device')
        self.previous[ids] = limit(velocity[..., :2].to(dtype=torch.float64), self.config.max_speed)

    def update(self, reference, reference_velocity, position, velocity, *, route_error=None,
               route_tangent=None, route_progress_error=None, route_mask=None):
        rp, rv, p, v = [self._xy(value) for value in
                        (reference, reference_velocity, position, velocity)]
        c = self.config
        raw = rv + c.kp*(rp-p) + c.kd*(rv-v)
        desired = limit(raw, c.max_speed)
        lateral = {}
        if route_error is not None or route_tangent is not None:
            error, tangent = self._xy(route_error), self._xy(route_tangent)
            norm = torch.linalg.vector_norm(tangent, dim=-1)
            torch._assert_async(torch.isclose(norm, torch.ones_like(norm), atol=1e-6, rtol=1e-5).all(),
                                'Route tangents must be unit vectors')
            normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=-1)
            cross_error = (error*normal).sum(-1)
            cross_velocity = (v*normal).sum(-1)
            cap = min(c.lateral_max_speed, c.max_speed)
            cross = (c.lateral_kp*cross_error-c.lateral_kd*cross_velocity).clamp(-cap, cap)
            budget = (c.max_speed**2-cross**2).clamp_min(0.).sqrt()
            along_raw = (raw*tangent).sum(-1)
            if route_progress_error is not None:
                progress = route_progress_error
                if (not isinstance(progress, torch.Tensor) or progress.device != p.device
                        or progress.shape != (len(p),)):
                    raise ValueError('Route progress error requires a per-agent device tensor')
                progress = progress.to(dtype=torch.float64)
                torch._assert_async(torch.isfinite(progress).all(), 'Route progress error must be finite')
                forward = torch.linalg.vector_norm(rv, dim=-1)
                along_raw = forward + c.kp*progress + c.kd*(forward-(v*tangent).sum(-1))
            along = torch.minimum(torch.maximum(along_raw, -budget), budget)
            lateral_desired = along[:, None]*tangent + cross[:, None]*normal
            desired = lateral_desired if route_mask is None else torch.where(route_mask[:, None], lateral_desired, desired)
            lateral = {'lateral_error': cross_error, 'lateral_velocity': cross_velocity,
                       'lateral_desired_speed': cross}
        delta = desired-self.previous
        self.previous.add_(limit(delta, c.max_acceleration*c.dt))
        self.report = {**lateral, 'position_error': rp-p, 'velocity_error': rv-v,
                       'speed_saturated': torch.linalg.vector_norm(raw, dim=-1) > c.max_speed,
                       'acceleration_saturated': torch.linalg.vector_norm(delta, dim=-1) > c.max_acceleration*c.dt}
        return self.previous.clone()
