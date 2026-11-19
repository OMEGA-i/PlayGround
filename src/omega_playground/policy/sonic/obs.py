"""SONIC tracker observations (``sonic.tracker_reference/1``, ``sonic.tracker_proprio/1``), batched, float64."""

import numpy as np
from scipy.spatial.transform import Rotation

from omega_playground.policy.base import MotionReference

ENCODER_DIM = 1751
FIELDS = ((0, 3), (3, 32), (32, 61), (61, 90), (90, 93))


def reference_window(ref: MotionReference, tick: int, frames: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Buffer frames tick, tick+stride, ... relative to each agent's t0; past the end the last frame holds."""
    if ref.dof_vel is None:
        raise ValueError("SONIC tracker needs MotionReference.dof_vel")
    start = np.rint(ref.t0.numpy() * ref.fps).astype(int)
    length = ref.length.numpy()
    idx = np.minimum(tick - start[:, None] + np.arange(frames)[None] * stride, length[:, None] - 1)
    rows = np.arange(len(start))[:, None]
    return ref.qpos.numpy()[rows, idx], ref.dof_vel.numpy()[rows, idx]


class TrackerObserver:
    """Encoder reference input and decoder proprio history (10 frames, field-major, repeat-first padding)."""

    def __init__(self, default_dof_pos, num_agents: int, frames: int = 10, mode_id: int = 0):
        self.default = np.asarray(default_dof_pos, np.float64)
        self.frames, self.mode_id = frames, mode_id
        dof = len(self.default)
        self.history = np.zeros((num_agents, frames, 6 + 3 * dof))
        self.filled = np.zeros(num_agents, bool)
        self.last_action = np.zeros((num_agents, dof), np.float32)

    def reset(self, agent_ids) -> None:
        self.filled[agent_ids] = False
        self.last_action[agent_ids] = 0.0

    def set_last_action(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, np.float32).copy()

    def __call__(self, root_rot_xyzw, root_ang_vel, dof_pos, dof_vel, ref_qpos, ref_dof_vel):
        """Root rotation xyzw and world angular velocity; joints in package (policy) order; ref [N, F, 7 + D]."""
        root = Rotation.from_quat(np.asarray(root_rot_xyzw, np.float64))
        row = np.concatenate([root.inv().apply(root_ang_vel), dof_pos - self.default, dof_vel, self.last_action,
                              root.inv().apply([0.0, 0.0, -1.0])], -1)
        self.history[~self.filled] = row[~self.filled, None]
        self.history[self.filled] = np.concatenate([self.history[self.filled, 1:], row[self.filled, None]], 1)
        self.filled[:] = True
        proprio = np.concatenate([self.history[:, :, a:b].reshape(len(row), -1) for a, b in FIELDS], -1)

        n, f = ref_qpos.shape[:2]
        forward = root.apply([1.0, 0.0, 0.0])
        heading = Rotation.from_euler("z", np.repeat(np.arctan2(forward[:, 1], forward[:, 0]), f)[:, None])
        relative = heading.inv() * Rotation.from_quat(ref_qpos[..., 3:7].reshape(-1, 4))
        enc = np.zeros((n, ENCODER_DIM), np.float32)
        enc[:, 0] = self.mode_id
        enc[:, 4:294] = ref_qpos[..., 7:].reshape(n, -1)
        enc[:, 294:584] = ref_dof_vel.reshape(n, -1)
        enc[:, 584:644] = relative.as_matrix().reshape(n, f, 3, 3)[..., :2].reshape(n, -1)
        return enc, proprio.astype(np.float32)
