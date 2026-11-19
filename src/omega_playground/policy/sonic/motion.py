"""SONIC release motion-buffer semantics in the release convention: rows [xyz, quat wxyz, joints in planner order]."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _rotations(q: np.ndarray) -> Rotation:
    return Rotation.from_quat(q[:, [4, 5, 6, 3]])


def sample_qpos(qpos: np.ndarray, times, fps: float) -> np.ndarray:
    """Linear interpolation, quaternion slerp; times outside the clip hold the endpoint."""
    t = np.clip(np.asarray(times), 0, (len(qpos) - 1) / fps)
    source = np.arange(len(qpos)) / fps
    out = np.stack([np.interp(t, source, qpos[:, i]) for i in range(qpos.shape[1])], -1)
    out[:, 3:7] = Slerp(source, _rotations(qpos))(t).as_quat()[:, [3, 0, 1, 2]]
    return out


def resample(qpos: np.ndarray, in_hz: float, out_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Planner frames -> buffer frames and joint velocities (forward difference, last row repeated)."""
    count = int(np.floor(len(qpos) / in_hz * out_hz))
    q = sample_qpos(qpos, np.arange(count) / out_hz, in_hz)
    velocity = np.diff(q[:, 7:], axis=0) * out_hz
    return q, np.r_[velocity, velocity[-1:]]


def blend(old_q: np.ndarray, old_v: np.ndarray, cursor: int, new_q: np.ndarray, new_v: np.ndarray,
          lookahead: int, ticks: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep ``lookahead`` frames of the old buffer from ``cursor``, then cross-fade to the new plan over ``ticks``."""
    indices = np.arange(lookahead + len(new_q))
    old_i = np.minimum(cursor + indices, len(old_q) - 1)
    new_i = np.clip(indices - lookahead, 0, len(new_q) - 1)
    alpha = np.clip((indices - lookahead) / ticks, 0, 1)[:, None]
    old, new = old_q[old_i], new_q[new_i]
    mixed = (1 - alpha) * old + alpha * new
    old_r = _rotations(old)
    delta = old_r.inv() * _rotations(new)
    mixed[:, 3:7] = (old_r * Rotation.from_rotvec(delta.as_rotvec() * alpha)).as_quat()[:, [3, 0, 1, 2]]
    return mixed, (1 - alpha) * old_v[old_i] + alpha * new_v[new_i]
