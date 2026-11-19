"""Device-resident sampling of an installed, time-parameterized route."""
import torch


class TensorReference:
    def __init__(self, reference, roles, device):
        def upload(value):
            return torch.tensor(value, device=device, dtype=torch.float64)
        self.xy, self.yaw, self.s = map(upload, (reference.xy, reference.yaw, reference.s))
        self.times, self.speed = map(upload, (reference.times, reference.speed))
        self.offsets = upload(reference.offsets)[roles]
        self.duration = reference.duration

    def distance_at(self, elapsed):
        if (not isinstance(elapsed, torch.Tensor) or elapsed.device != self.times.device
                or elapsed.shape != (len(self.offsets),)):
            raise ValueError('Elapsed times must be per-agent tensors on the route device')
        torch._assert_async(torch.isfinite(elapsed).all(), 'Elapsed times must be finite')
        t = elapsed.to(dtype=torch.float64).clamp(0., self.duration)
        i = (torch.searchsorted(self.times, t, right=True)-1).clamp(0, len(self.times)-2)
        tau = t-self.times[i]
        acceleration = (self.speed[i+1]-self.speed[i])/(self.times[i+1]-self.times[i])
        return self.s[i]+self.speed[i]*tau+.5*acceleration*tau.square()

    def time_at_distance(self, distance):
        if (not isinstance(distance, torch.Tensor) or distance.device != self.times.device
                or distance.shape != (len(self.offsets),)):
            raise ValueError('Distances must be per-agent tensors on the route device')
        torch._assert_async(torch.isfinite(distance).all(), 'Distances must be finite')
        distance = distance.to(dtype=torch.float64).clamp(self.s[0], self.s[-1])
        i = (torch.searchsorted(self.s, distance, right=True)-1).clamp(0, len(self.s)-2)
        ds = distance-self.s[i]
        acceleration = (self.speed[i+1]-self.speed[i])/(self.times[i+1]-self.times[i])
        end_speed = (self.speed[i].square()+2*acceleration*ds).clamp_min(0.).sqrt()
        return self.times[i]+2*ds/(self.speed[i]+end_speed).clamp_min(1e-12)

    def sample(self, elapsed):
        return self.sample_distance(self.distance_at(elapsed))

    def sample_distance(self, distance):
        distance = distance.clamp(self.s[0], self.s[-1])
        i = (torch.searchsorted(self.s, distance, right=True)-1).clamp(0, len(self.s)-2)
        fraction = (distance-self.s[i])/(self.s[i+1]-self.s[i]).clamp_min(1e-12)
        xy = self.xy[i]+fraction[:, None]*(self.xy[i+1]-self.xy[i])
        yaw = self.yaw[i]+fraction*(self.yaw[i+1]-self.yaw[i])
        forward, left = self.offsets.unbind(-1)
        shift = torch.stack((forward*yaw.cos()-left*yaw.sin(),
                             forward*yaw.sin()+left*yaw.cos()), dim=-1)
        return xy+shift, yaw

    def alignment_heading(self, elapsed, lookahead_m=.25):
        start, fallback = self.sample(elapsed)
        ahead, _ = self.sample_distance(self.distance_at(elapsed)+lookahead_m)
        delta = ahead-start
        return torch.where(torch.linalg.vector_norm(delta, dim=-1) > 1e-6,
                           torch.atan2(delta[:, 1], delta[:, 0]), fallback)
