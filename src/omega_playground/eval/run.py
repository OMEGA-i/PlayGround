"""Headless run on flat ground; writes run.json, metrics.json, trace.npz (and failure_* on a non-finite state)."""

import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from omega_playground.control.route_player import RoutePlayer
from omega_playground.control.teams import TaskControllerBatch
from omega_playground.policy.rotations import calc_heading
from omega_playground.policy.runner import WbcTick
from omega_playground.sim.world import PhysicsWorld

VERSIONS = ("warp-lang", "mujoco-warp", "mujoco", "torch", "numpy", "onnxruntime", "scipy")
TRACE_KEYS = ("sim_time", "root_pos", "root_quat_xyzw", "root_vel", "heading", "ground_height", "cmd_speed",
              "cmd_movement", "cmd_facing", "action", "pd_target", "dof_pos", "dof_vel")


def _git(*args) -> str:
    root = Path(__file__).resolve().parents[3]
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _versions() -> dict:
    out = {}
    for name in VERSIONS:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


class Recorder:
    def __init__(self, world: PhysicsWorld, output: Path, player=None):
        self.world, self.output, self.player = world, output, player
        self.trace = {k: [] for k in TRACE_KEYS}
        self.failure = None
        self.last_progress = -float("inf")

    def __call__(self, tick: WbcTick) -> bool:
        v, c = tick.view, tick.command
        values = (("sim_time", torch.full((v.root_pos.shape[0],), tick.sim_time)), ("root_pos", v.root_pos),
                  ("root_quat_xyzw", v.root_rot), ("root_vel", v.root_vel), ("heading", calc_heading(getattr(v, "heading_rot", v.root_rot))),
                  ("ground_height", v.ground_height),
                  ("cmd_speed", c.speed), ("cmd_movement", c.movement), ("cmd_facing", c.facing),
                  ("action", tick.action), ("pd_target", tick.pd_target), ("dof_pos", v.dof_pos),
                  ("dof_vel", v.dof_vel))
        for key, value in values:
            self.trace[key].append(value.detach().cpu().numpy())
        if tick.sim_time - self.last_progress >= 10. - 1e-9:
            self.last_progress = tick.sim_time
            (self.output / "progress.json").write_text(json.dumps({
                "sim_time": tick.sim_time, "physics_step": self.world.physics_steps,
                "root_pos": self.trace["root_pos"][-1].tolist(),
                "root_height": (self.trace["root_pos"][-1][:, 2] - self.trace["ground_height"][-1]).tolist(),
            }, indent=2) + "\n")
        if isinstance(self.player, (RoutePlayer, TaskControllerBatch)):
            for key, value in self.player.check(v, tick.sim_time).items():
                self.trace.setdefault(key, []).append(value.copy())
            if self.player.hold is not None:
                self.failure = self.player.hold
                (self.output / "failure.json").write_text(json.dumps(self.failure, indent=2) + "\n")
                np.savez(self.output / "failure_state.npz", **self.world.joint_state())
                return True
        if all(np.isfinite(self.trace[k][-1]).all() for k in ("root_pos", "dof_pos", "action")):
            return False
        self.failure = {"code": "nonfinite", "sim_time": tick.sim_time, "physics_step": self.world.physics_steps}
        (self.output / "failure.json").write_text(json.dumps(self.failure, indent=2) + "\n")
        np.savez(self.output / "failure_state.npz", **self.world.joint_state())
        return True

    def arrays(self) -> dict:
        return {k: np.stack(v) for k, v in self.trace.items() if v}


def summarize(a: dict, dt: float, fallen_height: float) -> dict:
    pos, vel = a["root_pos"], a["root_vel"]
    h = pos[..., 2] - a["ground_height"]
    last = slice(max(0, len(pos) - round(2.0 / dt)), None)
    agents = []
    for i in range(pos.shape[1]):
        speed = np.linalg.norm(vel[last, i, :2], axis=-1)
        move = a["cmd_movement"][last, i, :2]
        heading_err = np.angle(np.exp(1j * (a["heading"][last, i] - np.arctan2(move[:, 1], move[:, 0]))))
        face = a["cmd_facing"][last, i, :2]
        facing_err = np.angle(np.exp(1j * (a["heading"][last, i] - np.arctan2(face[:, 1], face[:, 0]))))
        fallen = np.flatnonzero(h[:, i] < fallen_height)
        agents.append({
            "displacement_xy": float(np.linalg.norm(pos[-1, i, :2] - pos[0, i, :2])),
            "path_length_xy": float(np.linalg.norm(np.diff(pos[:, i, :2], axis=0), axis=-1).sum()),
            "last2s_mean_speed": float(speed.mean()),
            "last2s_cmd_speed": float(a["cmd_speed"][last, i].mean()),
            "last2s_mean_abs_heading_error_deg": float(np.degrees(np.abs(heading_err)).mean()),
            "last2s_mean_abs_facing_error_deg": float(np.degrees(np.abs(facing_err)).mean()),
            "min_root_height": float(h[:, i].min()),
            "first_fallen_time": float(a["sim_time"][fallen[0], i]) if len(fallen) else None,
        })
    return {
        "min_root_height": float(h.min()),
        "fallen_frames": int((h < fallen_height).sum()),
        "max_abs_action": float(np.abs(a["action"]).max()),
        "agents": agents,
    }
