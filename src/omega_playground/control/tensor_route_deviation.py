"""Batched local route projection with independent device-resident cursors."""
import torch


class TensorRouteProjection:
    def __init__(self, reference, roles, device):
        # Route upload occurs once when a task/leg is installed, not each step.
        def tensor(value):
            return torch.tensor(value, device=device, dtype=torch.float64)
        import numpy as np
        from .route_deviation import LocalRouteProjection
        prepared = [LocalRouteProjection(reference, role) for role in roles]
        if not prepared or prepared[0].delta.shape[0] == 0:
            raise ValueError('Route projection requires agents and at least two samples')
        # Preserve the task's authoritative sampled geometry, including vertex
        # ties; CPU/GPU trig differences must not redefine the uploaded route.
        self.xy = tensor(np.stack([p.xy for p in prepared]))
        self.delta = tensor(np.stack([p.delta for p in prepared]))
        self.length2 = tensor(np.stack([p.length2 for p in prepared]))
        self.arc = tensor(np.stack([p.arc for p in prepared])).contiguous()
        self.reference_s = tensor(reference.s)
        self.progress = self.length2.new_zeros(len(roles))
        self.reference_distance = self.progress.clone()
        self.initial_tangent = tensor(np.stack([p.tangent for p in prepared]))
        self.tangent = self.initial_tangent.clone()
        self.indices = torch.arange(self.delta.shape[1], device=device)[None]

    def reset(self, ids):
        self.progress[ids] = 0.
        self.reference_distance[ids] = 0.
        self.tangent[ids] = self.initial_tangent[ids]

    def project(self, position, active=None):
        if (not isinstance(position, torch.Tensor) or position.device != self.xy.device
                or position.shape != self.tangent.shape):
            raise ValueError('Projection requires per-agent XY positions on the route device')
        if active is None:
            active = torch.ones(len(position), dtype=torch.bool, device=position.device)
        position = position.to(dtype=torch.float64)
        torch._assert_async(torch.isfinite(position).all(), 'Route positions must be finite')
        lo = (torch.searchsorted(self.arc, (self.progress-10.)[:, None]).squeeze(1)-1).clamp_min(0)
        hi = (torch.searchsorted(self.arc, (self.progress+10.)[:, None]).squeeze(1)+1).clamp_max(self.delta.shape[1])
        fraction = (((position[:, None]-self.xy[:, :-1])*self.delta).sum(-1)
                    / self.length2.clamp_min(1e-12)).clamp(0., 1.)
        projected = self.xy[:, :-1]+fraction[..., None]*self.delta
        distance2 = (projected-position[:, None]).square().sum(-1)
        local = (self.indices >= lo[:, None]) & (self.indices < hi[:, None])
        index = distance2.masked_fill(~local, float('inf')).argmin(-1)
        rows = torch.arange(len(position), device=position.device)
        f, length2 = fraction[rows, index], self.length2[rows, index]
        self.progress = torch.where(active & (distance2[rows, index] < 4.),
            self.arc[rows, index]+f*length2.sqrt(), self.progress)
        candidate = self.delta[rows, index]/length2.clamp_min(1e-12).sqrt()[:, None]
        self.tangent = torch.where((active & (length2 > 1e-12))[:, None], candidate, self.tangent)
        distance = (self.reference_s[index]
            + f*(self.reference_s[index+1]-self.reference_s[index]))
        self.reference_distance = torch.where(active, distance, self.reference_distance)
        return projected[rows, index]-position, self.reference_distance
