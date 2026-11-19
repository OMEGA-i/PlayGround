"""SONIC WBC: release encoder + decoder (ONNX, batch 1 per agent) tracking a ``MotionReference``."""

import numpy as np
import torch

from omega_playground.policy.base import MotionReference, RobotBatchView
from omega_playground.policy.package import PolicyPackage
from omega_playground.policy.schedule import Schedule
from omega_playground.policy.sonic import obs
from omega_playground.policy.sonic.planner import session


class SonicTracker:
    name = "sonic_tracker"
    condition = "motion_reference"

    def __init__(self, package: PolicyPackage, num_agents: int, device: torch.device):
        layer = package.manifest["layers"]["wbc"]
        modules = package.manifest["modules"]
        self.schedule = Schedule.from_json(layer["schedule"])
        window = layer["reference_window"]
        self.fps = package.manifest["layers"]["joystick"]["output"]["fps"]
        self.frames, self.stride = window["frames"], round(window["stride_s"] * self.fps)
        self.gpu = modules["encoder"]["runtime"].get("backend") == "torch_cuda_batch"
        if self.gpu:
            from .cuda_batch import CudaBatch
            from .tracker_graph import TrackerGraph
            torch.backends.cuda.matmul.allow_tf32 = False
            self.encoder = TrackerGraph(package.module_path("encoder")).to(device).eval()
            self.decoder = TrackerGraph(package.module_path("decoder")).to(device).eval()
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
            def forward(enc, proprio):
                token = self.encoder.batched(enc)
                return token, self.decoder.batched(torch.cat([token, proprio], -1))
            self.batch = CudaBatch(forward, num_agents, device)
        else:
            self.encoder, self.decoder = session(package, "encoder"), session(package, "decoder")
            for s, width in ((self.encoder, modules["encoder"]["inputs"]["tracker_reference"]["dim"]),
                             (self.decoder, modules["encoder"]["output"]["dim"]
                              + modules["decoder"]["inputs"]["tracker_proprio"]["dim"])):
                if len(s.get_inputs()) != 1 or s.get_inputs()[0].shape[-1] != width:
                    raise ValueError(f"ONNX input width differs from the package declaration ({width})")
        params = modules["encoder"]["inputs"]["tracker_reference"]["params"]
        history = modules["decoder"]["inputs"]["tracker_proprio"]["history"]["frames"]
        self.observer = obs.TrackerObserver(package.robot["default_dof_pos"], num_agents, history, params["mode_id"])
        self.device = device
        if self.gpu:
            from .torch_obs import DeviceReference, TorchTrackerObserver
            self.device_reference = DeviceReference(device)
            self.device_observer = TorchTrackerObserver(package.robot['default_dof_pos'], num_agents,
                history, params['mode_id'], device=device)
        if self.gpu:
            # Capture before the operation server becomes ready; no observer state is changed.
            self.infer(np.zeros((num_agents, obs.ENCODER_DIM), np.float32),
                       np.zeros((num_agents, modules["decoder"]["inputs"]["tracker_proprio"]["dim"]), np.float32))

    def reset(self, agent_ids: torch.Tensor) -> None:
        if self.gpu:
            self.device_observer.reset(agent_ids)
        else:
            self.observer.reset(agent_ids.cpu().numpy())

    def infer(self, enc: np.ndarray, proprio: np.ndarray, *, return_tokens=True) -> tuple[np.ndarray | None, np.ndarray]:
        if self.gpu:
            tokens, actions = self.batch(torch.as_tensor(enc, device=self.device),
                                         torch.as_tensor(proprio, device=self.device))
            tokens = tokens.cpu().numpy() if return_tokens else None
            actions = actions.cpu().numpy()
            if not np.isfinite(actions).all():
                raise RuntimeError("Nonfinite SONIC tracker action")
            return tokens, actions
        enc_name, dec_name = self.encoder.get_inputs()[0].name, self.decoder.get_inputs()[0].name
        tokens, actions = [], []
        for e, p in zip(enc, proprio, strict=True):
            token = self.encoder.run(None, {enc_name: e[None]})[0].reshape(-1)
            x = np.concatenate([token, p])[None].astype(np.float32)
            tokens.append(token)
            actions.append(self.decoder.run(None, {dec_name: x})[0].reshape(-1))
        actions = np.stack(actions)
        if not np.isfinite(actions).all():
            raise RuntimeError("Nonfinite SONIC tracker action")
        return np.stack(tokens), actions

    def act(self, view: RobotBatchView, condition: MotionReference, sim_time: float) -> torch.Tensor:
        tick = round(sim_time * self.fps)
        if self.gpu:
            from .torch_obs import reference_window
            q, v = reference_window(self.device_reference(condition), tick, self.frames, self.stride)
            enc, proprio = self.device_observer(view.root_rot, view.body_ang_vel[:, view.anchor],
                                                view.dof_pos, view.dof_vel, q, v)
            _, actions = self.batch(enc, proprio)
            torch._assert_async(torch.isfinite(actions).all(), 'Nonfinite SONIC tracker action')
            self.device_observer.set_last_action(actions)
            return actions
        ref_q, ref_v = obs.reference_window(condition, tick, self.frames, self.stride)

        def host(x):
            return x.detach().cpu().double().numpy()

        enc, proprio = self.observer(host(view.root_rot), host(view.body_ang_vel[:, view.anchor]),
                                     host(view.dof_pos), host(view.dof_vel), ref_q, ref_v)
        _, actions = self.infer(enc, proprio, return_tokens=False)
        self.observer.set_last_action(actions)
        return torch.from_numpy(actions).to(self.device)
