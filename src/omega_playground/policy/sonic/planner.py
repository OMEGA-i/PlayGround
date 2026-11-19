"""SONIC Joystick: release planner (ONNX) + 50 Hz motion buffer with the release replanning rules."""

from typing import NamedTuple

import numpy as np
import onnxruntime as ort
import torch

from omega_playground.policy.base import JoystickBatch, MotionReference, RobotBatchView
from omega_playground.policy.package import PolicyPackage
from omega_playground.policy.schedule import Schedule
from omega_playground.policy.sonic import motion, torch_motion


class Command(NamedTuple):
    mode: int
    speed: float
    height: float
    movement: tuple
    facing: tuple


def session(package: PolicyPackage, name: str) -> ort.InferenceSession:
    runtime = package.manifest["modules"][name]["runtime"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = runtime["intra_op_num_threads"]
    s = ort.InferenceSession(str(package.module_path(name)), options, providers=[runtime["provider"]])
    s.disable_fallback()
    return s


class SonicPlanner:
    def __init__(self, package: PolicyPackage, num_agents=1, device="cpu"):
        spec = package.manifest["modules"]["planner"]
        self.gpu = spec["runtime"].get("backend") == "torch_cuda_batch"
        if self.gpu:
            from .cuda_batch import CudaBatch
            from .planner_graph import TorchSPlanner
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.graph = TorchSPlanner(package.module_path("planner"), device=device, microbatch=num_agents)
            self.batch = CudaBatch(lambda *args: torch.vmap(self.graph)(*args), num_agents, device)
        else:
            self.session = session(package, "planner")
        self.seed = int(spec["inputs"]["random_seed"])
        self.tokens = np.asarray(spec["inputs"]["allowed_pred_num_tokens"], np.int64)
        self.min_frames, self.max_frames = spec["output"]["frames"]
        names = set(self.graph.inputs) if self.gpu else {x.name for x in self.session.get_inputs()}
        if names != set(spec["inputs"]):
            raise ValueError(f"Planner inputs {sorted(names)} differ from the package declaration")

        if self.gpu:
            # Warm/capture using a valid nominal pose, without publishing a motion buffer.
            rm = package.manifest["layers"]["joystick"]["release_motion"]
            c = rm["initial_command"]
            cmd = Command(c["mode"], c["speed"], c["height"], tuple(c["movement"]), tuple(c["facing"]))
            context = np.zeros((4, 36), np.float32)
            context[:, :3] = rm["initial_context_root_xyz"]
            context[:, 3] = 1
            joints = spec["qpos_layout"]["joint_names"]
            context[:, 7:] = np.asarray(package.robot["default_dof_pos"])[[package.dof_names.index(j) for j in joints]]
            self.plan_batch([context] * num_agents, [cmd] * num_agents)

    def values(self, context: np.ndarray, command: Command) -> dict:
        return {
            "context_mujoco_qpos": np.asarray(context, np.float32).reshape(1, 4, 36),
            "target_vel": np.array([command.speed], np.float32),
            "mode": np.array([command.mode], np.int64),
            "movement_direction": np.array([command.movement], np.float32),
            "facing_direction": np.array([command.facing], np.float32),
            "random_seed": np.array([self.seed], np.int64),
            "height": np.array([command.height], np.float32),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": self.tokens,
        }

    def plan(self, context: np.ndarray, command: Command) -> np.ndarray:
        return self.plan_batch([context], [command])[0]

    def plan_batch(self, contexts, commands, *, return_tensors=False):
        if not len(contexts) or len(contexts) != len(commands):
            raise ValueError('Planner needs one context per command')
        if self.gpu:
            # Commands are still host orchestration. Context and generated poses
            # remain on device for the live Joystick path.
            rows = [self.values(np.zeros((4, 36), np.float32), cmd) for cmd in commands]
            values = [torch.stack([torch.as_tensor(c, device=self.graph.device, dtype=torch.float32)
                                   .reshape(1, 4, 36) for c in contexts])
                      if k == 'context_mujoco_qpos' else
                      torch.as_tensor(np.stack([r[k] for r in rows]), device=self.graph.device)
                      for k in self.graph.inputs]
            outputs = dict(zip(self.graph.outputs, self.batch(*values), strict=True))
            if return_tensors:
                # Only variable-length metadata crosses to host, never pose data.
                counts = outputs['num_pred_frames'].reshape(-1).cpu().tolist()
                result = []
                for i, count in enumerate(counts):
                    count = int(count)
                    if not self.min_frames <= count <= self.max_frames:
                        raise ValueError('Invalid planner output length')
                    q = outputs['mujoco_qpos'][i, 0, :count]
                    if q.shape != (count, 36):
                        raise ValueError('Invalid planner output shape')
                    torch._assert_async(torch.isfinite(q).all(), 'Invalid planner output')
                    # CUDA graph output storage is overwritten on the next replay.
                    result.append(q.clone())
                return result
            host = {k: v.cpu().numpy() for k, v in outputs.items()}
            return [self.validate({k: v[i] for k, v in host.items()}) for i in range(len(rows))]
        if return_tensors:
            raise RuntimeError('Device planner output requires torch_cuda_batch')
        rows = [self.values(c, cmd) for c, cmd in zip(contexts, commands, strict=True)]
        return [self.validate(dict(zip((x.name for x in self.session.get_outputs()),
                                      self.session.run(None, v), strict=True))) for v in rows]

    def plan_tensors(self, contexts, commands):
        """Fixed-shape output and per-row valid lengths; no host readback."""
        if not self.gpu:
            raise RuntimeError('Tensor planner requires torch_cuda_batch')
        n, device = len(contexts), self.graph.device
        values = {
            'context_mujoco_qpos': contexts.float()[:, None],
            'target_vel': commands.speed.float()[:, None],
            'mode': commands.mode.long()[:, None],
            'movement_direction': commands.movement.float()[:, None],
            'facing_direction': commands.facing.float()[:, None],
            'random_seed': torch.full((n, 1), self.seed, dtype=torch.long, device=device),
            'height': commands.height.float()[:, None],
            'has_specific_target': torch.zeros(n, 1, 1, dtype=torch.long, device=device),
            'specific_target_positions': torch.zeros(n, 1, 4, 3, device=device),
            'specific_target_headings': torch.zeros(n, 1, 4, device=device),
            'allowed_pred_num_tokens': torch.as_tensor(self.tokens, device=device).unsqueeze(0).expand(n, *self.tokens.shape),
        }
        out = dict(zip(self.graph.outputs, self.batch(*(values[k] for k in self.graph.inputs)), strict=True))
        lengths = out['num_pred_frames'].reshape(-1).long()
        torch._assert_async(((lengths >= self.min_frames) & (lengths <= self.max_frames)).all(),
                            'Invalid planner frame count')
        q = out['mujoco_qpos'][:, 0, :self.max_frames]
        torch._assert_async(torch.isfinite(q).all(), 'Invalid planner output')
        return q.clone(), lengths.clone()

    def validate(self, out):
        count = int(np.asarray(out["num_pred_frames"]).item())
        qpos = out["mujoco_qpos"][0, :count]
        if not self.min_frames <= count <= self.max_frames or qpos.shape != (count, 36) or not np.isfinite(qpos).all():
            raise ValueError("Invalid planner output")
        return qpos


class SonicJoystick:
    """Per-agent release motion buffers; replans only on poll ticks, using the outgoing plan as context."""

    name = "sonic_planner"

    def __init__(self, package: PolicyPackage, num_agents: int, device: torch.device):
        layer = package.manifest["layers"]["joystick"]
        rm = layer["release_motion"]
        self.schedule = Schedule.from_json(layer["schedule"])
        self.hz = layer["output"]["fps"]
        self.planner_hz = rm["planner_output_hz"]
        self.poll_ticks = round(self.schedule.interval_s() * self.hz)
        if abs(self.schedule.interval_s() * self.hz - self.poll_ticks) > 1e-9:
            raise ValueError("Joystick poll must be a whole number of buffer frames")
        self.lookahead, self.blend_ticks = rm["lookahead_ticks"], rm["blend_ticks"]
        self.context_frames = rm["context_frames"]
        self.static_modes = set(rm["static_modes"])
        self.poll_count = rm["replan_poll_count"]
        c = rm["initial_command"]
        self.initial = Command(int(c["mode"]), float(c["speed"]), float(c["height"]), tuple(c["movement"]),
                               tuple(c["facing"]))
        self.root_xyz = np.asarray(rm["initial_context_root_xyz"], np.float64)
        joints = package.manifest["modules"]["planner"]["qpos_layout"]["joint_names"]
        dofs = package.dof_names
        self.to_package = np.array([joints.index(d) for d in dofs])
        self.to_planner = np.array([dofs.index(j) for j in joints])
        self.planner = SonicPlanner(package, num_agents, device)
        self.robot_id, self.num_agents, self.device = package.robot["id"], num_agents, device
        self.qpos: list[np.ndarray | torch.Tensor | None] = [None] * num_agents
        self.vel: list[np.ndarray | torch.Tensor | None] = [None] * num_agents
        self.start = [0] * num_agents
        self.polls = [0] * num_agents
        self.last: list[Command | None] = [None] * num_agents
        self.replans = 0
        self.device_buffers = None
        if type(self) is SonicJoystick and self.planner.gpu:
            from .device_joystick import DeviceJoystick
            self.device_buffers = DeviceJoystick(self)

    def reset(self, agent_ids: torch.Tensor, view: RobotBatchView) -> None:
        if self.device_buffers is not None:
            return self.device_buffers.reset(agent_ids)
        for i in agent_ids.tolist():
            self.qpos[i] = self.vel[i] = self.last[i] = None
            self.polls[i] = 0

    def _tick(self, sim_time: float) -> int:
        tick = round(sim_time * self.hz)
        if abs(sim_time * self.hz - tick) > 1e-6:
            raise ValueError(f"sim_time {sim_time} is not on the {self.hz} Hz buffer grid")
        return tick

    @staticmethod
    def commands(c: JoystickBatch) -> list[Command]:
        return [Command(int(m), float(s), float(h), tuple(mv), tuple(f)) for m, s, h, mv, f in
                zip(c.mode.tolist(), c.speed.tolist(), c.height.tolist(), c.movement.tolist(), c.facing.tolist(),
                    strict=True)]

    def should_replan(self, commands: JoystickBatch, sim_time: float) -> torch.Tensor:
        tick = self._tick(sim_time)
        if self.device_buffers is not None:
            return self.device_buffers.should_replan(commands, tick)
        mask = []
        for i, cmd in enumerate(self.commands(commands)):
            if self.qpos[i] is None:
                mask.append(True)
                continue
            if tick == 0 or tick % self.poll_ticks:
                mask.append(False)
                continue
            self.polls[i] += 1
            timed = self.polls[i] >= self.poll_count.get(str(cmd.mode), self.poll_count["default"])
            if timed:
                self.polls[i] = 0
            old = self.last[i]
            changed = (cmd.mode, cmd.facing, cmd.height) != (old.mode, old.facing, old.height)
            moving = cmd.mode not in self.static_modes and (
                cmd.movement != old.movement or cmd.speed != old.speed or (timed and cmd.speed != 0))
            mask.append(changed or moving)
        return torch.tensor(mask, dtype=torch.bool)

    def _current(self, view: RobotBatchView, i: int):
        if self.planner.gpu:
            return torch.cat((view.root_pos[i].double(), view.root_rot[i, [3, 0, 1, 2]].double(),
                              view.dof_pos[i, self.to_planner].double())).to(self.device)
        pos = view.root_pos[i].detach().cpu().double().numpy()
        rot = view.root_rot[i].detach().cpu().double().numpy()[[3, 0, 1, 2]]
        dof = view.dof_pos[i].detach().cpu().double().numpy()[self.to_planner]
        return np.r_[pos, rot, dof]

    def plan(self, view: RobotBatchView, commands: JoystickBatch, agent_ids: torch.Tensor,
             sim_time: float) -> MotionReference:
        tick = self._tick(sim_time)
        if self.device_buffers is not None:
            return self.device_buffers.plan(view, commands, agent_ids, tick)
        cmds = self.commands(commands)
        ids = agent_ids.tolist()
        contexts, selected = [], []
        for i in ids:
            if self.qpos[i] is None:
                cmd = self.initial
                current = self._current(view, i)
                if self.planner.gpu:
                    context = torch.as_tensor(current, device=self.device).repeat(self.context_frames, 1)
                    context[:, :3] = torch.as_tensor(self.root_xyz, device=self.device)
                else:
                    context = np.tile(current, (self.context_frames, 1))
                    context[:, :3] = self.root_xyz
            else:
                cmd, cursor = cmds[i], tick - self.start[i]
                times = (cursor + self.lookahead) / self.hz + np.arange(self.context_frames) / self.planner_hz
                context = (torch_motion if self.planner.gpu else motion).sample_qpos(self.qpos[i], times, self.hz)
            contexts.append(context)
            selected.append(cmd)
        if self.planner.gpu:
            predictions = self.planner.plan_batch(contexts, selected, return_tensors=True)
            # Group equal output lengths; resample all agents in a group together.
            groups = {}
            for row, prediction in enumerate(predictions):
                groups.setdefault(len(prediction), []).append(row)
            resampled = {}
            for rows in groups.values():
                q, v = torch_motion.resample(torch.stack([predictions[r] for r in rows]),
                                              self.planner_hz, self.hz)
                resampled.update({r: (q[j], v[j]) for j, r in enumerate(rows)})
        else:
            predictions = self.planner.plan_batch(contexts, selected)
            resampled = {r: motion.resample(p, self.planner_hz, self.hz) for r, p in enumerate(predictions)}
        for row, (i, cmd) in enumerate(zip(ids, selected, strict=True)):
            new_q, new_v = resampled[row]
            if self.qpos[i] is None:
                q, v = new_q, new_v
                self.polls[i] = 0
            else:
                q, v = (torch_motion if self.planner.gpu else motion).blend(self.qpos[i], self.vel[i], tick - self.start[i], new_q, new_v,
                                    self.lookahead, self.blend_ticks)
            self.qpos[i], self.vel[i], self.start[i], self.last[i] = q, v, tick, cmd
            self.replans += 1
        return self.reference()

    def reference(self) -> MotionReference:
        if self.device_buffers is not None:
            return self.device_buffers.reference()
        if any(q is None for q in self.qpos):
            raise RuntimeError("Every agent needs a plan before the reference is published")
        k = max(len(q) for q in self.qpos)
        if self.planner.gpu:
            poses, velocities = [], []
            for q, v in zip(self.qpos, self.vel, strict=True):
                rows = torch.cat((q[:, :3], q[:, [4, 5, 6, 3]], q[:, 7:][:, self.to_package]), -1)
                poses.append(torch.cat((rows, rows[-1:].expand(k-len(rows), -1))))
                reordered = v[:, self.to_package]
                velocities.append(torch.cat((reordered, reordered[-1:].expand(k-len(v), -1))))
            return MotionReference(self.robot_id,
                torch.tensor(self.start, dtype=torch.float64, device=self.device)/self.hz,
                float(self.hz), torch.stack(poses),
                torch.tensor([len(q) for q in self.qpos], device=self.device), torch.stack(velocities))
        qpos = np.zeros((self.num_agents, k, 36))
        vel = np.zeros((self.num_agents, k, len(self.to_package)))
        for i, (q, v) in enumerate(zip(self.qpos, self.vel, strict=True)):
            rows = np.concatenate([q[:, :3], q[:, [4, 5, 6, 3]], q[:, 7:][:, self.to_package]], -1)
            qpos[i, :len(q)], qpos[i, len(q):] = rows, rows[-1]
            vel[i, :len(v)], vel[i, len(v):] = v[:, self.to_package], v[-1, self.to_package]
        return MotionReference(
            robot_id=self.robot_id,
            t0=torch.tensor(self.start, dtype=torch.float64) / self.hz,
            fps=float(self.hz),
            qpos=torch.from_numpy(qpos),
            length=torch.tensor([len(q) for q in self.qpos]),
            dof_vel=torch.from_numpy(vel),
        )
