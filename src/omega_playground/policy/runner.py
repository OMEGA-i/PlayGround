"""Simulation-time scheduler (contracts §3.1): Player -> Joystick -> WBC -> action transform -> PD -> physics."""

import time
import os
from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor

from omega_playground.policy import registry
from omega_playground.policy.action import ActionTransform
from omega_playground.policy.base import JoystickBatch, MotionReference, Player, RobotBatchView
from omega_playground.policy.package import PolicyPackage


class StageTimer:
    def __init__(self, cuda: bool):
        self.cuda, self.samples = cuda, {}
        self.pending_events = {}
        self.trace_scopes = bool(os.environ.get("OMEGA_PHYSX_CUDA_TRACE"))

    def __call__(self, name: str):
        timer = self

        class _Scope:
            def __enter__(self):
                self.trace = torch.profiler.record_function("playground::"+name) if timer.trace_scopes else None
                if self.trace is not None:
                    self.trace.__enter__()
                self.events = None
                if timer.cuda:
                    self.events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    self.events[0].record()
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timer.samples.setdefault(name, []).append(time.perf_counter() - self.t)
                if self.events is not None:
                    self.events[1].record()
                    timer.pending_events.setdefault(name, []).append(self.events)
                if self.trace is not None:
                    self.trace.__exit__(*exc)

        return _Scope()

    def report(self) -> dict:
        # Only reporting waits for GPU completion, never the control hot path.
        if self.pending_events:
            torch.cuda.synchronize()
        result = {}
        for name, host in self.samples.items():
            events = self.pending_events.get(name)
            values = [a.elapsed_time(b)/1000 for a, b in events] if events else host
            result[name] = {"count": len(values), "mean_ms": 1e3*float(np.mean(values)),
                            "p95_ms": 1e3*float(np.percentile(values, 95)), "total_s": float(np.sum(values)),
                            "measurement": "cuda_events" if events else "host_wall",
                            "host_mean_ms": 1e3*float(np.mean(host))}
        return result


@dataclass(frozen=True)
class WbcTick:
    sim_time: float
    view: RobotBatchView
    command: JoystickBatch
    reference: MotionReference | None
    action: Tensor
    pd_target: Tensor


class Runner:
    def __init__(self, world, package: PolicyPackage, player: Player, device: torch.device,
                 on_wbc: Callable[[WbcTick], bool | None] | None = None, task=None, joystick_factory=None):
        if world.physics_hz != package.physics_hz:
            raise ValueError(f"world runs at {world.physics_hz} Hz, package needs {package.physics_hz} Hz")
        self.world, self.package, self.player, self.task, self.on_wbc = world, package, player, task, on_wbc
        self.device = device
        n = world.num_agents
        self.wbc = registry.create_wbc(package, n, device)
        self.joystick = (joystick_factory or registry.create_joystick)(package, n, device)
        try:
            if (self.joystick is None) != (self.wbc.condition == "joystick"):
                raise ValueError(f"WBC condition {self.wbc.condition!r} does not match the Joystick layer")
            self.action = ActionTransform(package.manifest["action"], device)
            hz = package.physics_hz
            self.player_every = player.schedule.physics_steps(hz)
            self.joystick_every = self.joystick.schedule.physics_steps(hz) if self.joystick else None
            self.wbc_every = self.wbc.schedule.physics_steps(hz)
            stride = getattr(world, 'step_stride', 1)
            if any(period is not None and period % stride for period in
                   (self.player_every, self.joystick_every, self.wbc_every)):
                raise ValueError('Policy cadence must align with the GPU physics step stride')
            self.timer = StageTimer(device.type == "cuda")
            self.command: JoystickBatch | None = None
            self.reference: MotionReference | None = None
            self.stopped = False
            self.pending_reset = torch.zeros(n, dtype=torch.bool, device=device)
            self.has_pending_reset = False
        except BaseException:
            self.close()
            raise

    def close(self):
        """Release policy-owned workers before rebuilding or exiting the simulator."""
        joystick = self.joystick
        self.joystick = None
        close = getattr(joystick, 'close', None)
        if close is not None:
            close()

    def request_reset(self, agent_ids):
        """Refresh selected planner buffers on the next WBC tick, including off-period holds."""
        self.pending_reset[agent_ids] = True
        self.has_pending_reset = self.has_pending_reset or len(agent_ids) > 0

    def rates(self) -> dict:
        hz = self.package.physics_hz
        return {"physics_hz": hz, "player_hz": hz / self.player_every, "wbc_hz": hz / self.wbc_every,
                "joystick_poll_hz": hz / self.joystick_every if self.joystick else None}

    def step(self) -> None:
        k, t = self.world.physics_steps, self.world.sim_time
        view = None

        def read():
            nonlocal view
            if view is None:
                with self.timer("state_read"):
                    # Keep the physics tensor view on its device. Any remaining
                    # host-only adapter owns its explicit compatibility boundary.
                    view = self.world.robot_view()
            return view

        if getattr(self, 'keyboard', None) is not None:
            if k % self.joystick_every == 0 or self.command is None:
                self.command = self.keyboard.act(read())
        elif k % self.player_every == 0:
            with self.timer("player"):
                self.command = self.player.act(read(), self.task, t)
        refresh = (self.pending_reset.nonzero().flatten()
                   if self.has_pending_reset and k % self.wbc_every == 0 else [])
        if len(refresh) and k % self.player_every != 0:
            self.command = JoystickBatch(**{f.name: getattr(self.command, f.name).clone() for f in fields(JoystickBatch)})
            self.command.mode[refresh] = 0
            self.command.speed[refresh] = -1
            self.command.movement[refresh] = 0
            state = read()
            quat = getattr(state, "heading_rot", state.root_rot)[refresh.to(state.root_rot.device)]
            x, y, z, w = quat.unbind(-1)
            yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
            self.command.facing[refresh] = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), dim=-1).to(self.command.facing)
        if self.joystick is not None and (k % self.joystick_every == 0 or len(refresh)):
            with self.timer("joystick"):
                mask = (self.joystick.should_replan(self.command, t).to(self.pending_reset.device)
                        if k % self.joystick_every == 0 else torch.zeros_like(self.pending_reset))
                mask[refresh] = True
                ids = mask.nonzero().flatten()
                if len(ids):
                    self.reference = self.joystick.plan(read(), self.command, ids, t)
        if k % self.wbc_every == 0:
            condition = self.command if self.wbc.condition == "joystick" else self.reference
            with self.timer("wbc"):
                raw = self.wbc.act(read(), condition, t)
                target = self.action(raw)
            if self.has_pending_reset:
                self.pending_reset[:] = False
                self.has_pending_reset = False
            self.world.set_pd_targets(target)
            if self.on_wbc is not None and self.on_wbc(WbcTick(t, view, self.command, self.reference, raw, target)):
                self.stopped = True
                return
        with self.timer("physics"):
            self.world.step()

    def run(self, physics_steps: int) -> None:
        for _ in range(physics_steps):
            self.step()
            if self.stopped:
                return
