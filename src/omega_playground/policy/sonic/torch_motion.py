"""Device-resident release motion buffers: xyz, quaternion wxyz, planner joints.

Interpolation is float64 to preserve the SciPy buffer contract. Leading batch
rows are independent; no tensor values are read back to the host.
"""
import math
from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _time_grid(count, fps, device):
    # CUDA scalar division may use reciprocal multiplication, moving an exact
    # keyframe by one ulp and selecting the opposite quaternion sign. Cache
    # the release's IEEE host time constants once; all pose arithmetic is GPU.
    return torch.tensor([i/fps for i in range(count)], dtype=torch.float64, device=device)


def slerp(a, b, alpha):
    a = a / torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    b = b / torch.linalg.vector_norm(b, dim=-1, keepdim=True)
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(max=1.)
    angle = torch.acos(dot)
    sine = torch.sin(angle).clamp_min(1e-15)
    spherical = (torch.sin((1-alpha)*angle)*a + torch.sin(alpha*angle)*b) / sine
    linear = (1-alpha)*a + alpha*b
    result = torch.where(dot > 1-1e-10, linear, spherical)
    return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True)


def sample_qpos(qpos, times, fps):
    """Sample [K,D] or [N,K,D]; times [T] or [N,T], endpoint hold."""
    if not math.isfinite(fps) or fps <= 0 or qpos.shape[-2] < 1:
        raise ValueError('Motion sampling requires positive fps and nonempty frames')
    single = qpos.ndim == 2
    q = qpos.double().unsqueeze(0) if single else qpos.double()
    if q.ndim != 3 or q.shape[-1] < 7:
        raise ValueError('Expected [K,7+D] or [N,K,7+D] motion')
    t = torch.as_tensor(times, dtype=torch.float64, device=q.device)
    if t.ndim == 1:
        t = t.unsqueeze(0).expand(len(q), -1)
    if t.ndim != 2 or len(t) != len(q):
        raise ValueError('Times must be shared [T] or batched [N,T]')
    t = t.clamp(0, (q.shape[1]-1)/fps)
    source = _time_grid(q.shape[1], fps, q.device)
    # SciPy Slerp selects the preceding segment at exact keyframes. This also
    # preserves its quaternion sign convention when adjacent keys are antipodal.
    left = (torch.searchsorted(source, t.contiguous(), right=False)-1).clamp(0, max(q.shape[1]-2, 0))
    right = (left+1).clamp_max(q.shape[1]-1)
    alpha = ((t-source[left])*fps)[..., None]
    rows = torch.arange(len(q), device=q.device)[:, None]
    a, b = q[rows, left], q[rows, right]
    result = a + alpha*(b-a)
    result[..., 3:7] = slerp(a[..., 3:7], b[..., 3:7], alpha)
    return result[0] if single else result


def resample(qpos, in_hz, out_hz):
    """Batch resampling and forward joint differences, with repeated final row."""
    if not all(math.isfinite(hz) and hz > 0 for hz in (in_hz, out_hz)):
        raise ValueError('Resampling rates must be positive and finite')
    count = math.floor(qpos.shape[-2]/in_hz*out_hz)
    if count < 2:
        raise ValueError('Resampled motion needs at least two frames for joint velocity')
    times = _time_grid(count, out_hz, qpos.device)
    q = sample_qpos(qpos, times, in_hz)
    velocity = torch.diff(q[..., 7:], dim=-2)*out_hz
    return q, torch.cat((velocity, velocity[..., -1:, :]), dim=-2)


def blend(old_q, old_v, cursor, new_q, new_v, lookahead, ticks):
    indices = torch.arange(lookahead+len(new_q), device=new_q.device)
    old_i = (cursor+indices).clamp_max(len(old_q)-1)
    new_i = (indices-lookahead).clamp(0, len(new_q)-1)
    alpha = ((indices-lookahead).double()/ticks).clamp(0, 1)[:, None]
    old, new = old_q[old_i], new_q[new_i]
    mixed = (1-alpha)*old + alpha*new
    mixed[:, 3:7] = slerp(old[:, 3:7], new[:, 3:7], alpha)
    return mixed, (1-alpha)*old_v[old_i] + alpha*new_v[new_i]
