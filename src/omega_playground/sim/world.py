"""Physics boundary shared by policy scheduling, operations and rendering.

Body order is package order, repeated per agent. Rotations are XYZW, positions
are world-space metres, and body velocities refer to link origins (not COMs).
Backend-native articulation buffers must not escape through this interface.
"""
from typing import Protocol

import numpy as np
import torch

from omega_playground.policy.base import RobotBatchView


class PhysicsWorld(Protocol):
    num_agents: int
    physics_hz: int
    physics_steps: int
    anchor: int

    @property
    def sim_time(self) -> float: ...

    def robot_view(self) -> RobotBatchView: ...
    def set_pd_targets(self, targets: torch.Tensor) -> None: ...
    def step(self) -> None: ...
    def body_q_snapshot(self) -> np.ndarray: ...
    def reset_agents(self, agent_ids, spawn_xy=None, yaw=None) -> dict: ...
    def apply_root_velocity_delta(self, agent_ids, delta_world: torch.Tensor) -> None: ...
    def joint_state(self) -> dict[str, np.ndarray]: ...
    def describe(self) -> dict: ...


def body_pose_snapshot(view: RobotBatchView) -> np.ndarray:
    """Renderer state independent of PhysX/Newton body and quaternion layouts."""
    return torch.cat((view.body_pos, view.body_rot), dim=-1).reshape(-1, 7).detach().cpu().numpy().copy()
