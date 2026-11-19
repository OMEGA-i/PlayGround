"""GPU contact/finite-state warnings; warnings never pause the workbench."""

import numpy as np
import torch
import warp as wp


class MJWarpSafety:
    def __init__(self, world):
        self.world = world
        self.nonfoot_count = np.zeros(1, dtype=int)
        self.samples = 0
        m = world.cpu_model
        self.geom_body = torch.as_tensor(m.geom_bodyid, device=world.torch_device, dtype=torch.long)
        self.nonfoot = torch.tensor(
            [
                bool(i and m.body(i).name not in ("left_ankle_roll_link", "right_ankle_roll_link"))
                for i in m.geom_bodyid
            ],
            device=world.torch_device,
        )
        self.indices = torch.arange(world.data.naconmax, device=world.torch_device)

    def sample(self):
        w = self.world
        d = w.data
        self.samples += 1
        pairs = wp.to_torch(d.contact.geom).long().clamp(0, len(self.geom_body) - 1)
        active = (self.indices < wp.to_torch(d.nacon)[0]) & (wp.to_torch(d.contact.dist) <= 0)
        a, b = pairs.unbind(-1)
        nonfoot = (
            active
            & (((self.geom_body[a] == 0) & self.nonfoot[b]) | ((self.geom_body[b] == 0) & self.nonfoot[a]))
        ).any()
        view = w.robot_view()
        height = view.root_pos[0, 2] - view.ground_height[0]
        finite = torch.isfinite(w.view("qpos")).all() & torch.isfinite(w.view("qvel")).all()
        up_z = 1 - 2 * view.root_rot[0, :2].square().sum()
        values = (
            torch.stack(
                (
                    height,
                    finite.float(),
                    nonfoot.float(),
                    wp.to_torch(d.nacon)[0].float(),
                    w.view("nefc")[0].float(),
                    up_z,
                )
            )
            .cpu()
            .numpy()
        )
        self.nonfoot_count[:] = self.nonfoot_count + 1 if values[2] else 0
        reason = None
        if not values[1]:
            reason = {"code": "nonfinite"}
        elif values[3] >= d.naconmax or values[4] >= d.njmax:
            reason = {"code": "physics_capacity", "contacts": int(values[3]), "constraints": int(values[4])}
        elif not np.isfinite(values[0]) or values[0] < 0.5 or values[5] < 0.5:
            reason = {"code": "fallen_in_formation", "agents": [0], "height_m": [float(values[0])]}
        elif self.nonfoot_count[0] >= 3:
            # Native G1 calf/ankle collision meshes can touch while upright.
            # Keep the contact warning, but do not label this as a fall.
            reason = {"code": "nonfoot_contact", "agents": [0], "height_m": [float(values[0])]}
        if reason:
            reason.update(sim_time=w.sim_time, physics_step=w.physics_steps)
        return reason

    def report(self):
        return {
            "physics_samples": self.samples,
            "backend": "mujoco_warp",
            "contact_definition": "robot non-foot / scene manifold separation <= 0",
            "action": "warn",
        }
