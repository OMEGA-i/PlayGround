"""Device-resident batched SONIC observations, matching the release NumPy oracle."""
from dataclasses import replace

import torch

from .obs import ENCODER_DIM, FIELDS


def normalized(q):
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-15)


def rotate(q, v):
    xyz, w = q[..., :3], q[..., 3:]
    return v + 2 * torch.cross(xyz, torch.cross(xyz, v, dim=-1) + w*v, dim=-1)


def conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def multiply(a, b):
    return torch.cat((a[..., 3:]*b[..., :3] + b[..., 3:]*a[..., :3]
                      + torch.cross(a[..., :3], b[..., :3], dim=-1),
                      a[..., 3:]*b[..., 3:] - (a[..., :3]*b[..., :3]).sum(-1, keepdim=True)), -1)


def columns(q):
    x, y, z, w = normalized(q).unbind(-1)
    # Row-major flatten of matrix[..., :2], matching scipy Rotation.as_matrix.
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*y+z*w),
                        1-2*(x*x+z*z), 2*(x*z-y*w), 2*(y*z+x*w)), -1)


class DeviceReference:
    """Upload a newly published host plan once, never on every tracker tick."""
    def __init__(self, device):
        self.device = device
        self.source = self.value = None
        self.versions = None

    def __call__(self, reference):
        tensors = (reference.t0, reference.length, reference.qpos, reference.dof_vel)
        # Inference tensors do not have version counters; publishers must replace
        # the MotionReference on each publication (the runtime contract).
        versions = tuple(None if t is None else (id(t), None if t.is_inference() else t._version) for t in tensors)
        if reference is not self.source or versions != self.versions:
            self.value = replace(reference, **{name: getattr(reference, name).to(self.device)
                for name in ('t0', 'length', 'qpos', 'dof_vel') if getattr(reference, name) is not None})
            self.source, self.versions = reference, versions
        return self.value


def reference_window(ref, tick, frames, stride):
    if ref.dof_vel is None:
        raise ValueError('SONIC tracker needs MotionReference.dof_vel')
    device = ref.qpos.device
    start = torch.round(ref.t0 * ref.fps).long()
    idx = tick-start[:, None] + torch.arange(frames, device=device)[None]*stride
    idx = idx.clamp_min(0).minimum(ref.length[:, None]-1)
    rows = torch.arange(len(start), device=device)[:, None]
    return ref.qpos[rows, idx], ref.dof_vel[rows, idx]


class TorchTrackerObserver:
    def __init__(self, default, count, frames=10, mode_id=0, *, device, a3=False):
        self.default = torch.as_tensor(default, dtype=torch.float64, device=device)
        self.history = torch.zeros(count, frames, 6+3*len(default), dtype=torch.float64, device=device)
        self.filled = torch.zeros(count, dtype=torch.bool, device=device)
        self.last_action = torch.zeros(count, len(default), dtype=torch.float32, device=device)
        self.mode_id, self.a3 = mode_id, a3

    def reset(self, ids):
        self.filled[ids] = False
        self.last_action[ids] = 0

    def set_last_action(self, action):
        self.last_action.copy_(action)

    def __call__(self, root_rotation, angular_velocity, positions, velocities, ref_qpos, ref_vel):
        root = normalized(root_rotation.to(torch.float64))
        inverse = conjugate(root)
        gravity = torch.zeros_like(angular_velocity, dtype=torch.float64)
        gravity[..., 2] = -1
        row = torch.cat((rotate(inverse, angular_velocity.double()), positions.double()-self.default,
                         velocities.double(), self.last_action.double(), rotate(inverse, gravity)), -1)
        shifted = torch.cat((self.history[:, 1:], row[:, None]), 1)
        self.history.copy_(torch.where(self.filled[:, None, None], shifted, row[:, None]))
        self.filled.fill_(True)
        proprio = torch.cat([self.history[:, :, a:b].flatten(1) for a, b in FIELDS], -1).float()
        n, f = ref_qpos.shape[:2]
        if self.a3:
            base = inverse[:, None].expand(-1, f, -1)
        else:
            forward = torch.zeros_like(gravity)
            forward[..., 0] = 1
            forward = rotate(root, forward)
            yaw = torch.atan2(forward[..., 1], forward[..., 0])
            heading = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw),
                                   -torch.sin(yaw/2), torch.cos(yaw/2)), -1)
            base = heading[:, None].expand(-1, f, -1)
        orientation = columns(multiply(base, normalized(ref_qpos[..., 3:7].double()))).flatten(1)
        if self.a3:
            return torch.cat((ref_qpos[..., 7:].flatten(1), ref_vel.flatten(1), orientation, proprio), -1).float()
        enc = torch.zeros(n, ENCODER_DIM, device=root.device, dtype=torch.float32)
        enc[:, 0] = self.mode_id
        enc[:, 4:294] = ref_qpos[..., 7:].flatten(1)
        enc[:, 294:584] = ref_vel.flatten(1)
        enc[:, 584:644] = orientation
        return enc, proprio
