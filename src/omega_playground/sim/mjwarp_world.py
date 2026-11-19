"""Single-agent GPU MuJoCo world using the policy's native named joints and PD."""

from types import SimpleNamespace

import mujoco
import mujoco_warp as mw
import numpy as np
import torch
import warp as wp

from omega_playground.ops.prepared_assets import checked_metadata
from omega_playground.policy.base import RobotBatchView

NAN = wp.constant(float("nan"))


@wp.kernel
def torque_pd(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    qa: wp.array[int],
    va: wp.array[int],
    kp: wp.array[float],
    kd: wp.array[float],
    limit: wp.array[float],
    target: wp.array2d[float],
    forces: wp.array2d[float],
):
    j = wp.tid()
    forces[0, va[j]] = wp.clamp(
        kp[j] * (target[0, j] - qpos[0, qa[j]]) - kd[j] * qvel[0, va[j]], -limit[j], limit[j]
    )


@wp.kernel
def ground_ray(mesh: wp.uint64, xy: wp.array2d[float], tops: wp.array[float], height: wp.array[float]):
    i = wp.tid()
    top = tops[i]
    query = wp.mesh_query_ray(mesh, wp.vec3(xy[i, 0], xy[i, 1], top), wp.vec3(0.0, 0.0, -1.0), 10000.0)
    height[i] = top - query.t if query.result else NAN


class MeshGround:
    def __init__(self, model, device):
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        vertices, triangles, count = [], [], 0
        for g in range(model.ngeom):
            if model.geom_bodyid[g] or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError("Prepared static collisions must be convex meshes")
            mesh = model.geom_dataid[g]
            start, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            v = model.mesh_vert[start : start + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            start, n = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            triangles.append(model.mesh_face[start : start + n] + count)
            vertices.append(v)
            count += len(v)
        vertices = np.concatenate(vertices).astype(np.float32)
        self.mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=device),
            indices=wp.array(np.concatenate(triangles).ravel(), dtype=int, device=device),
        )
        self.top = float(vertices[:, 2].max() + 1)
        self.device = device

    def ground_heights(self, points, below=None):
        xy = points[:, :2].contiguous()
        tops = (
            (points[:, 2] if points.shape[1] > 2 else xy.new_full((len(xy),), self.top))
            if below is None
            else below
        )
        tops = tops.contiguous()
        out = torch.empty(len(xy), device=xy.device)
        wp.launch(
            ground_ray,
            dim=len(xy),
            inputs=[self.mesh.id, wp.from_torch(xy), wp.from_torch(tops)],
            outputs=[wp.from_torch(out)],
            device=self.device,
        )
        return out

    def heights(self, xy, below=None, *, allow_missing=False):
        points = torch.as_tensor(np.asarray(xy), dtype=torch.float32, device=self.device)
        tops = None if below is None else torch.as_tensor(below, dtype=torch.float32, device=self.device)
        result = self.ground_heights(points, tops).cpu().numpy()
        if not allow_missing and not np.isfinite(result).all():
            raise ValueError("No authored ground under position")
        return result

    def describe(self):
        return {"backend": "warp_static_mesh", "device": str(self.device)}


class MJWarpWorld:
    backend = "mujoco_warp"
    step_stride = 4

    def __init__(self, package, asset, scene, spawn, device):
        from pathlib import Path

        if len(spawn) != 1 or not str(device).startswith("cuda"):
            raise ValueError("MJWarp workbench requires one agent on CUDA")
        meta = checked_metadata(Path(asset))
        if (
            meta["schema"] != "omega.mjwarp-physics/1"
            or meta["policy_sha256"] != package.sha256sum()
            or meta["scene_sha256"] != scene.scene_sha256
            or meta["collision_sha256"] != scene.collision_sha256
        ):
            raise ValueError("Prepared MJWarp physics contract mismatch")
        if meta["mujoco_version"] != mujoco.__version__:
            raise ValueError("Prepared MJB requires its compiling MuJoCo version")
        self.package, self.num_agents, self.torch_device = package, 1, torch.device(device)
        self.physics_hz, self.physics_steps = package.physics_hz, 0
        self.physics_dt = 1 / self.physics_hz
        m = self.cpu_model = mujoco.MjModel.from_binary_path(str(Path(asset) / meta["entry"]))
        if abs(m.opt.timestep - self.physics_dt) > 1e-9:
            raise ValueError("Physics timestep mismatch")
        self.bodies = torch.tensor([m.body(n).id for n in package.body_names], device=device)
        self.anchor = package.body_names.index(package.robot["anchor_body"])
        self.bridge = SimpleNamespace(heading_offset=0.0)
        jids = np.array([m.joint(n).id for n in package.dof_names])
        self.qids = torch.tensor(m.jnt_qposadr[jids], device=device)
        self.vids = torch.tensor(m.jnt_dofadr[jids], device=device)
        self.ground = MeshGround(m, device)
        self.model = mw.put_model(m)
        self.data = mw.make_data(m, nworld=1, nconmax=4096, njmax=4096)
        self.target = torch.tensor([package.robot["default_dof_pos"]], device=device)
        pd = package.manifest["pd"]
        self.pd = [
            wp.array(m.jnt_qposadr[jids], dtype=int, device=device),
            wp.array(m.jnt_dofadr[jids], dtype=int, device=device),
        ]
        self.pd += [wp.array(pd[key], dtype=float, device=device) for key in ("kp", "kd", "effort_limit")]
        self.initial_xy = np.asarray(spawn)[:, :2].copy()
        self.reset_agents([0], self.initial_xy)

        def advance():
            wp.launch(
                torque_pd,
                dim=len(jids),
                inputs=[self.data.qpos, self.data.qvel, *self.pd, wp.from_torch(self.target)],
                outputs=[self.data.qfrc_applied],
                device=device,
            )
            mw.step(self.model, self.data)

        advance()
        self.reset_agents([0], self.initial_xy)
        with wp.ScopedCapture() as capture:
            for _ in range(4):
                advance()
            # Refresh poses and link velocities at the integrated timestamp.
            # Contacts already come from the last substep; a fifth solve is unnecessary.
            mw.kinematics(self.model, self.data)
            mw.com_pos(self.model, self.data)
            mw.com_vel(self.model, self.data)
        self.graph = capture.graph
        self.rootids = torch.as_tensor(m.body_rootid, device=device, dtype=torch.long)

    @property
    def sim_time(self):
        return self.physics_steps / self.physics_hz

    def view(self, name):
        return wp.to_torch(getattr(self.data, name))

    def robot_view(self):
        pos = self.view("xpos")
        cv = self.view("cvel")
        linear = cv[..., 3:] + torch.cross(
            cv[..., :3], pos - self.view("subtree_com")[:, self.rootids], dim=-1
        )
        return RobotBatchView(
            pos[:, self.bodies],
            self.view("xquat")[:, self.bodies][:, :, [1, 2, 3, 0]],
            linear[:, self.bodies],
            cv[:, self.bodies, :3],
            self.view("qpos")[:, self.qids],
            self.view("qvel")[:, self.vids],
            self.ground.ground_heights(pos[:, self.bodies[self.anchor]]),
            self.anchor,
        )

    def set_pd_targets(self, targets):
        self.target.copy_(targets)

    def step(self):
        wp.capture_launch(self.graph)
        self.physics_steps += 4

    def body_q_tensor(self):
        return torch.cat(
            (self.view("xpos")[:, self.bodies], self.view("xquat")[:, self.bodies][:, :, [1, 2, 3, 0]]), -1
        ).reshape(-1, 7)

    def body_q_snapshot(self):
        return self.body_q_tensor().cpu().numpy().copy()

    def reset_agents(self, agent_ids, spawn_xy=None, yaw=None):
        if list(agent_ids) != [0]:
            raise ValueError("Only agent zero exists")
        xy = self.initial_xy if spawn_xy is None else np.asarray(spawn_xy)
        z = self.ground.heights(xy)[0] + self.package.robot["default_root_height"]
        q = torch.as_tensor(self.cpu_model.qpos0.copy(), device=self.torch_device).float()[None]
        q[0, :3] = torch.tensor([*xy[0], z], device=self.torch_device)
        angle = 0.0 if yaw is None else float(yaw[0])
        q[0, 3:7] = torch.tensor([np.cos(angle / 2), 0, 0, np.sin(angle / 2)], device=self.torch_device)
        q[:, self.qids] = torch.tensor(self.package.robot["default_dof_pos"], device=self.torch_device)
        mw.reset_data(self.model, self.data)
        self.view("qpos").copy_(q)
        self.target.copy_(q[:, self.qids])
        mw.forward(self.model, self.data)
        return {"agents": [0]}

    def joint_state(self):
        v = self.robot_view()
        return {
            "joint_q": self.view("qpos").cpu().numpy().copy(),
            "joint_qd": self.view("qvel").cpu().numpy().copy(),
            "body_q": self.body_q_snapshot(),
            "body_qd": torch.cat((v.body_vel, v.body_ang_vel), -1).reshape(-1, 6).cpu().numpy(),
        }

    def describe(self):
        return {
            "backend": "mujoco_warp",
            "physics_device": str(self.torch_device),
            "physics_dt": self.physics_dt,
            "num_agents": 1,
            "cuda_graph": True,
            "substeps_per_launch": 4,
            "ground": self.ground.describe(),
            "body_order": self.package.body_names,
            "dof_order": self.package.dof_names,
        }
