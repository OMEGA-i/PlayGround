"""Quaternion helpers (xyzw) with the exact arithmetic ASE was trained with."""

import torch
from torch import Tensor


def normalize(x: Tensor, eps: float = 1e-9) -> Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    shape = a.shape
    a, b = a.reshape(-1, 4), b.reshape(-1, 4)
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=-1).reshape(shape)


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    flat_q = q.reshape(-1, 4)
    flat_v = v.reshape(-1, 3)
    q_w, q_vec = flat_q[:, 3], flat_q[:, :3]
    a = flat_v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, flat_v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.reshape(-1, 1, 3), flat_v.reshape(-1, 3, 1)).squeeze(-1) * 2.0
    return (a + b + c).reshape(v.shape)


def quat_from_angle_axis(angle: Tensor, axis: Tensor) -> Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    return normalize(torch.cat([xyz, theta.cos()], dim=-1))


def calc_heading(q: Tensor) -> Tensor:
    ref = torch.zeros_like(q[..., :3])
    ref[..., 0] = 1
    d = quat_rotate(q, ref)
    return torch.atan2(d[..., 1], d[..., 0])


def calc_heading_quat_inv(q: Tensor) -> Tensor:
    axis = torch.zeros_like(q[..., :3])
    axis[..., 2] = 1
    return quat_from_angle_axis(-calc_heading(q), axis)


def quat_to_tan_norm(q: Tensor) -> Tensor:
    ref_tan = torch.zeros_like(q[..., :3])
    ref_tan[..., 0] = 1
    ref_norm = torch.zeros_like(q[..., :3])
    ref_norm[..., 2] = 1
    return torch.cat([quat_rotate(q, ref_tan), quat_rotate(q, ref_norm)], dim=-1)
