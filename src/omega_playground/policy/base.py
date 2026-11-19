"""Player -> Joystick -> WBC interfaces (contracts §3.1). All methods are batched over agents."""

from dataclasses import dataclass, fields
from typing import Any, Literal, Protocol

from torch import Tensor

from omega_playground.policy.schedule import Schedule


@dataclass(frozen=True)
class RobotBatchView:
    """Read-only robot state in package order; quaternions xyzw; velocities of body origins, world frame."""

    body_pos: Tensor        # [N, B, 3]
    body_rot: Tensor        # [N, B, 4]
    body_vel: Tensor        # [N, B, 3]
    body_ang_vel: Tensor    # [N, B, 3]
    dof_pos: Tensor         # [N, D]
    dof_vel: Tensor         # [N, D]
    ground_height: Tensor   # [N] under the anchor body
    anchor: int = 0
    heading_offset: float = 0.0  # Anatomical forward yaw relative to the physical root X axis.

    def to(self, device):
        """Move only state tensors across the physics/inference boundary."""
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        moved = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in values.items()}
        if all(moved[key] is value for key, value in values.items()):
            return self
        return RobotBatchView(**moved)

    @property
    def root_pos(self) -> Tensor:
        return self.body_pos[:, self.anchor]

    @property
    def root_rot(self) -> Tensor:
        return self.body_rot[:, self.anchor]

    @property
    def heading_rot(self) -> Tensor:
        if self.heading_offset == 0.0:
            return self.root_rot
        import math

        from omega_playground.policy.rotations import quat_mul
        offset = self.root_rot.new_tensor([0., 0., math.sin(self.heading_offset/2), math.cos(self.heading_offset/2)])
        return quat_mul(self.root_rot, offset.expand_as(self.root_rot))

    @property
    def root_vel(self) -> Tensor:
        return self.body_vel[:, self.anchor]


@dataclass(frozen=True)
class JoystickBatch:
    """Batched ``JoystickCommand`` (contracts §6.1); directions are world-frame unit vectors."""

    mode: Tensor            # [N] int
    speed: Tensor           # [N]
    movement: Tensor        # [N, 3], z = 0
    facing: Tensor          # [N, 3], z = 0
    vertical_speed: Tensor  # [N]
    height: Tensor          # [N], -1 = unspecified


@dataclass(frozen=True)
class MotionReference:
    """Joystick -> WBC: time-stamped reference motion; endpoint hold past the last frame."""

    robot_id: str
    t0: Tensor              # [N] sim time of frame 0
    fps: float
    qpos: Tensor            # [N, K, 7 + D]: root xyz, root quat xyzw, joints in package dof order
    length: Tensor          # [N] valid frames per agent
    dof_vel: Tensor | None = None  # [N, K, D] joint velocities when the producer defines them (blended with qpos)
    schema: str = "omega.motion_ref/1"


class Player(Protocol):
    """One robot: target trajectory + own state -> movement intent.

    FFPD is the NPC implementation. Squad coordination and batching are separate.
    """
    name: str
    schedule: Schedule

    def reset(self, agent_ids: Tensor, view: RobotBatchView) -> None: ...

    def act(self, view: RobotBatchView, task: Any, sim_time: float) -> JoystickBatch: ...


class Joystick(Protocol):
    name: str
    schedule: Schedule

    def reset(self, agent_ids: Tensor, view: RobotBatchView) -> None: ...

    def should_replan(self, commands: JoystickBatch, sim_time: float) -> Tensor: ...

    def plan(self, view: RobotBatchView, commands: JoystickBatch, agent_ids: Tensor,
             sim_time: float) -> MotionReference: ...


class WbcPolicy(Protocol):
    name: str
    schedule: Schedule
    condition: Literal["motion_reference", "joystick"]

    def reset(self, agent_ids: Tensor) -> None: ...

    def act(self, view: RobotBatchView, condition: "MotionReference | JoystickBatch",
            sim_time: float) -> Tensor: ...
