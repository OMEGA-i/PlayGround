"""Fixed-input parity: recompute every fixture stage with OmegaPlayground's own code."""

import numpy as np
import torch

from omega_playground.policy.action import ActionTransform
from omega_playground.policy.package import PolicyPackage


def _sonic(package: PolicyPackage, f: dict, device: torch.device) -> dict:
    from omega_playground.policy.base import JoystickBatch, RobotBatchView
    from omega_playground.policy.sonic import motion, obs
    from omega_playground.policy.sonic.planner import Command, SonicJoystick
    from omega_playground.policy.sonic.tracker import SonicTracker

    sequences, steps = f["tracker_action"].shape[:2]
    tracker = SonicTracker(package, sequences, device)
    out = {k: [] for k in ("enc", "proprio", "token", "action")}
    for t in range(steps):
        enc, proprio = tracker.observer(*(f[f"tracker_{k}"][:, t] for k in (
            "root_rot_xyzw", "root_ang_vel", "dof_pos", "dof_vel", "ref_qpos", "ref_dof_vel")))
        token, action = tracker.infer(enc, proprio)
        tracker.observer.set_last_action(action)
        for k, v in zip(out, (enc, proprio, token, action), strict=True):
            out[k].append(v)
    out = {k: np.stack(v, 1) for k, v in out.items()}
    act = ActionTransform(package.manifest["action"], device)

    joystick = SonicJoystick(package, 1, device)
    planned = {k: [] for k in ("count", "qpos", "resampled_count", "resampled_qpos", "resampled_vel")}
    for i in range(len(f["planner_count"])):
        cmd = Command(int(f["planner_mode"][i]), float(f["planner_speed"][i]), float(f["planner_height"][i]),
                      tuple(f["planner_movement"][i].tolist()), tuple(f["planner_facing"][i].tolist()))
        qpos = joystick.planner.plan(f["planner_context"][i], cmd)
        q, v = motion.resample(qpos, joystick.planner_hz, joystick.hz)
        for key, value, rows in (("qpos", qpos, f["planner_qpos"].shape[1]),
                                 ("resampled_qpos", q, f["planner_resampled_qpos"].shape[1]),
                                 ("resampled_vel", v, f["planner_resampled_vel"].shape[1])):
            padded = np.zeros((rows, value.shape[1]))
            padded[:len(value)] = value
            planned[key].append(padded)
        planned["count"].append(len(qpos))
        planned["resampled_count"].append(len(q))

    init = torch.from_numpy(f["sequence_initial_qpos"])
    zeros = torch.zeros(1, 1, 3, dtype=torch.float64)
    view = RobotBatchView(body_pos=init[None, None, :3], body_rot=init[None, None, 3:7], body_vel=zeros,
                          body_ang_vel=zeros, dof_pos=init[None, 7:], dof_vel=torch.zeros_like(init[None, 7:]),
                          ground_height=torch.zeros(1))
    joystick = SonicJoystick(package, 1, device)
    refs, vels, replanned, reference = [], [], [], None
    for tick, c in enumerate(f["sequence_commands"]):
        t = tick / joystick.hz
        cmd = JoystickBatch(mode=torch.tensor([int(c[0])]), speed=torch.tensor(c[1:2]),
                            movement=torch.tensor(c[None, 3:6]), facing=torch.tensor(c[None, 6:9]),
                            vertical_speed=torch.zeros(1), height=torch.tensor(c[2:3]))
        flag = False
        if tick % joystick.poll_ticks == 0:
            ids = joystick.should_replan(cmd, t).nonzero().flatten()
            if len(ids):
                reference, flag = joystick.plan(view, cmd, ids, t), True
        q, v = obs.reference_window(reference, tick, tracker.frames, tracker.stride)
        refs.append(q[0])
        vels.append(v[0])
        replanned.append(flag)

    return {
        "tracker_encoder_obs": (out["enc"], "observation"),
        "tracker_proprio_obs": (out["proprio"], "observation"),
        "tracker_token": (out["token"], "module"),
        "tracker_action": (out["action"], "module"),
        "tracker_pd_target": (act(torch.from_numpy(f["tracker_action"]).to(device)), "pd_target"),
        **{f"planner_{k}": (np.asarray(v), "motion_velocity" if k == "resampled_vel" else "planner")
           for k, v in planned.items()},
        "sequence_ref_qpos": (np.stack(refs), "motion_reference"),
        "sequence_ref_dof_vel": (np.stack(vels), "motion_velocity"),
        "sequence_replanned": (np.asarray(replanned, np.float64), "motion_reference"),
    }


def verify(package: PolicyPackage, device: str = "cpu") -> dict:
    parity = package.manifest["parity"]
    fixtures = dict(np.load(package.root / parity["file"]))
    tolerance = {k: v["atol"] for k, v in parity["tolerance"].items()}
    tolerance.setdefault("chain", tolerance["module"])
    if package.family != 'sonic' or not package.robot['id'].startswith('unitree_g1'):
        raise ValueError('SONIC verification requires a G1 package')
    computed = _sonic(package, fixtures, torch.device(device))
    checks = {}
    for name, (value, kind) in computed.items():
        expected = fixtures[name]
        value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        diff = float(np.abs(value.astype(np.float64) - expected.astype(np.float64)).max())
        checks[name] = {"max_abs_diff": diff, "atol": tolerance[kind], "passed": diff <= tolerance[kind]}
    return {
        "package": package.name,
        "package_sha256sums": package.sha256sum(),
        "device": device,
        "fixtures": int(parity["count"]),
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
    }
