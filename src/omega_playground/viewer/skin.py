"""Render-only skinned avatars (UE4 Mannequin, UE5 Manny) driven by ASE body poses; never touches simulation state.

Ported from ProtoMotions-ASE-newton ``utils/ase_manny.py`` (Epic weights, bind matrices and UVs exported by Blender).
Body rotations are xyzw and positions are body origins. The calibration is specific to the ASE skeleton.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation, Slerp

TEXTURE_CHANNELS = ("basecolor", "normal", "roughness", "metallic")


@dataclass(frozen=True)
class SkinPackage:
    root: Path
    meta: dict

    @property
    def name(self) -> str:
        return self.meta["name"]

    def describe(self) -> dict:
        return {"name": self.name, "appearance": self.meta["appearance"], "root": str(self.root)}


def load(root: str | Path) -> SkinPackage:
    root = Path(root)
    meta = json.loads((root / "skin.json").read_text())
    for rel, digest in json.loads((root / "SHA256SUMS.json").read_text()).items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f"skin file {rel} does not match SHA256SUMS.json")
    return SkinPackage(root, meta)


@wp.kernel
def _skin(points: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3), joints: wp.array2d(dtype=wp.int32),
          weights: wp.array2d(dtype=wp.float32), palette: wp.array(dtype=wp.mat44), count: int,
          output: wp.array(dtype=wp.vec3), output_normals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    p = wp.vec3(0.0)
    n = wp.vec3(0.0)
    for j in range(count):
        matrix = palette[joints[i, j]]
        p += weights[i, j] * wp.transform_point(matrix, points[i])
        n += weights[i, j] * wp.transform_vector(matrix, normals[i])
    output[i] = p
    output_normals[i] = wp.normalize(n)


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cross = np.cross(a, b)
    dot = np.clip(a @ b, -1, 1)
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            return np.eye(3)
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis)).as_matrix()
    return Rotation.from_rotvec(np.arccos(dot) * cross / np.linalg.norm(cross)).as_matrix()


class MannyPalette:
    """Per-bone skinning matrices from ASE body poses."""

    def __init__(self, data, body_names: list[str], robot_xml: Path):
        self.names = data["names"].tolist()
        self.bind = data["bind"].astype(np.float64)
        self.parents = data["parents"]
        self.authored_neck_chain = str(data.get("appearance", "")) == "UE4 Mannequin"
        self.chest_bone = max((n for n in self.names if n.startswith("spine_")),
                              key=lambda n: self.bind[self.names.index(n), 2, 3])
        # Blender Manny faces -Y with left +X. ASE faces +X with left +Y.
        self.basis = basis = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        self.bind[:, :3, :3] = basis @ self.bind[:, :3, :3]
        self.bind[:, :3, 3] = self.bind[:, :3, 3] @ basis.T
        self.inverse = np.linalg.inv(self.bind)
        self.body_names = list(body_names)
        self.rest = {}

        def visit(node, origin):
            origin = origin + np.fromstring(node.get("pos", "0 0 0"), sep=" ")
            if node.get("quat") or node.get("euler"):
                raise ValueError("ASE calibration expects identity rest body rotations")
            self.rest[node.get("name")] = origin
            for child in node.findall("body"):
                visit(child, origin)

        for root in ET.parse(robot_xml).findall("./worldbody/body"):
            visit(root, np.zeros(3))
        self.mapping = {"pelvis": "pelvis", "head": "head"}
        ends = {}
        for side, word in (("l", "left"), ("r", "right")):
            for bone, body, child in (("upperarm", "upper_arm", "lowerarm"), ("lowerarm", "lower_arm", "hand"),
                                      ("hand", "hand", None), ("thigh", "thigh", "calf"),
                                      ("calf", "shin", "foot"), ("foot", "foot", None)):
                self.mapping[f"{bone}_{side}"] = f"{word}_{body}"
                if child:
                    ends[f"{bone}_{side}"] = f"{child}_{side}"
        self.sole_offsets = {}
        for side in ("l", "r"):
            foot, toe = self.names.index(f"foot_{side}"), self.names.index(f"ball_{side}")
            heights = []
            for slot in range(int(data["slots"])):
                mask = (data[f"s{slot}_weights"] * np.isin(data[f"s{slot}_joints"], [foot, toe])).sum(1) > 0.8
                heights.extend(data[f"s{slot}_points"][mask, 2])
            self.sole_offsets[f"foot_{side}"] = self.bind[foot, 2, 3] - min(heights) - 0.05
        self.calibration = {}
        for bone, body in self.mapping.items():
            i = self.names.index(bone)
            if bone in ends:
                source = self.bind[self.names.index(ends[bone]), :3, 3] - self.bind[i, :3, 3]
                delta = _align(source, self.rest[self.mapping[ends[bone]]] - self.rest[body])
            elif bone.startswith("hand"):
                owner = bone.replace("hand", "lowerarm")
                delta = self.calibration[owner] @ self.bind[self.names.index(owner), :3, :3].T
            else:
                delta = np.eye(3)
            self.calibration[bone] = delta @ self.bind[i, :3, :3]

        # Palm grip frames (hand body coordinates) and finger curls that close the hands on the props.
        self.grips, self.grip_rotations = {}, {}
        for side, sign in (("r", 1), ("l", -1)):
            hand = self.names.index(f"hand_{side}")
            delta = self.calibration[f"hand_{side}"] @ self.bind[hand, :3, :3].T
            knuckles = np.array([delta @ (self.bind[self.names.index(f"{finger}_01_{side}"), :3, 3]
                                          - self.bind[hand, :3, 3]) for finger in ("index", "middle", "ring", "pinky")])
            axis = knuckles[0] - knuckles[-1]
            axis /= np.linalg.norm(axis)
            distal = knuckles.mean(0)
            distal -= axis * np.dot(distal, axis)
            distal /= np.linalg.norm(distal)
            volar = np.cross(axis, distal) * sign
            self.grips[side] = (knuckles.mean(0) + 0.025 * volar + 0.007 * distal, axis)
            curls = [(f"{finger}_{joint:02d}_{side}", angle) for finger in ("index", "middle", "ring", "pinky")
                     for joint, angle in ((1, 48), (2, 78), (3, 55))]
            # Thumb opposition toward the handle, then a modest distal curl.
            curls += [(f"thumb_{joint:02d}_{side}", angle) for joint, angle in ((1, -15), (2, 20), (3, 40))]
            for name, angle in curls:
                local = self.bind[self.names.index(name), :3, :3].T @ delta.T @ axis
                self.grip_rotations[name] = Rotation.from_rotvec(local * np.deg2rad(sign * angle)).as_matrix()

    def matrices(self, positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
        rotations = Rotation.from_quat(quaternions).as_matrix()
        state = {name: (positions[i], rotations[i]) for i, name in enumerate(self.body_names)}
        targets = np.tile(np.eye(4), (len(self.names), 1, 1))
        pelvis_z = self.bind[self.names.index("pelvis"), 2, 3]
        chest_z = self.bind[self.names.index(self.chest_bone), 2, 3]
        for i, name in enumerate(self.names):
            parent = int(self.parents[i])
            if parent >= 0:
                targets[i] = targets[parent] @ self.inverse[parent] @ self.bind[i]
            else:
                p, r = state["pelvis"]
                targets[i, :3, :3] = r @ self.bind[i, :3, :3]
                targets[i, :3, 3] = p + r @ (self.bind[i, :3, 3] - self.rest["pelvis"])
            if name in self.grip_rotations:
                targets[i, :3, :3] = targets[i, :3, :3] @ self.grip_rotations[name]
            if name in self.mapping:
                p, r = state[self.mapping[name]]
                targets[i, :3, :3] = r @ self.calibration[name]
                # UE4 has one longer neck bone: keep its parent-relative bind translation.
                if not (name == "head" and self.authored_neck_chain):
                    targets[i, :3, 3] = p
                if name == "head" and not self.authored_neck_chain:
                    targets[i, :3, 3] += r @ np.array([0, 0, 0.10])
                elif name in self.sole_offsets:
                    targets[i, :3, 3] += r @ np.array([0, 0, self.sole_offsets[name]])
            elif name.startswith("spine_"):
                f = np.clip((self.bind[i, 2, 3] - pelvis_z) / (chest_z - pelvis_z), 0, 1)
                p0, r0 = state["pelvis"]
                p1, r1 = state["torso"]
                r = Slerp([0, 1], Rotation.from_matrix([r0, r1]))([f]).as_matrix()[0]
                targets[i, :3, :3] = r @ self.bind[i, :3, :3]
                targets[i, :3, 3] = (1 - f) * p0 + f * (p1 + r1 @ np.array([0, 0, 0.19]))
            elif name.startswith("neck_"):
                p, r = state["head"]
                targets[i, :3, :3] = r @ self.bind[i, :3, :3]
                if not self.authored_neck_chain:
                    targets[i, :3, 3] = p + r @ np.array([0, 0, 0.03 if name.endswith("01") else 0.065])
        return targets @ self.inverse


class SkinRenderer:
    """Skins one avatar per agent from host-side ``body_q`` and logs the meshes into an RTX viewer."""

    def __init__(self, skin: SkinPackage, body_names: list[str], robot_xml: Path, num_agents: int, device, *, optimized=True, armed=True, equipment_path=None):
        robot_sha = hashlib.sha256(Path(robot_xml).read_bytes()).hexdigest()
        if skin.meta["skeleton_sha256"] != robot_sha:
            raise ValueError(f"skin {skin.name} is calibrated for skeleton {skin.meta['skeleton_sha256'][:12]}, "
                             f"the policy robot is {robot_sha[:12]}")
        self.skin, self.device, self.num_agents = skin, device, num_agents
        self.optimized = optimized
        soma = skin.meta.get("kind") == "soma23"
        self.native = skin.meta.get("kind", "manny") == "native_robot" or (soma and skin.meta['rigid_robot'])
        if soma:
            from omega_playground.viewer.armed_skin import load_soma_grip
            from omega_playground.viewer.soma_skin import Soma23Palette
            data = load_soma_grip(skin.root) if armed else np.load(skin.root / "skin.npz")
            self.adapter = Soma23Palette(data, body_names, robot_xml)
            self.adapter.basis = np.eye(3)
        elif self.native:
            from omega_playground.viewer.native_skin import NativeASEPalette

            robot = skin.meta["robot"]
            data = np.load(skin.root / "robot_skin.npz")
            if armed:
                prepared = skin.root / "armed_robot_skin.npz"
                if not prepared.is_file():
                    raise ValueError('Armed skin missing; run tools/assets/prepare_armed_assets.py')
                source_sha = hashlib.sha256((skin.root / "robot_skin.npz").read_bytes()).hexdigest()
                data = np.load(prepared)
                if str(data['source_sha256']) != source_sha:
                    raise ValueError('Armed skin is stale; run tools/assets/prepare_armed_assets.py')
            self.adapter = NativeASEPalette(data, body_names, robot)
        else:
            data = np.load(skin.root / "manny.npz")
            self.adapter = MannyPalette(data, body_names, Path(robot_xml))
            if not armed:
                self.adapter.grip_rotations.clear()
        self.sole = None
        if not soma:
            from omega_playground.viewer.soma_skin import SoleSupport
            feet = ('left_foot', 'right_foot') if self.native else ('foot_l', 'ball_l', 'foot_r', 'ball_r')
            self.sole = SoleSupport(data, body_names, robot_xml,
                                    bones=[self.adapter.names.index(n) for n in feet],
                                    support_bodies=('left_foot', 'right_foot'), basis=self.adapter.basis)
        self.bodies = len(body_names)
        self.palettes = [wp.zeros(len(self.adapter.names), dtype=wp.mat44, device=device) for _ in range(num_agents)]

        def arr(slot, key, dtype):
            return wp.array(data[f"s{slot}_{key}"], dtype=dtype, device=device)

        def rotated(slot, key):
            # Vertices share the bind matrices' frame (authored skin frame -> ASE heading).
            return wp.array((data[f"s{slot}_{key}"] @ self.adapter.basis.T).astype(np.float32), dtype=wp.vec3,
                            device=device)

        self.parts = []
        self.rigid_bones = []
        for slot in range(int(data["slots"])):
            weights, joints = data[f"s{slot}_weights"], data[f"s{slot}_joints"]
            used = np.unique(joints[weights != 0])
            self.rigid_bones.append(int(used[0]) if len(used) == 1 and np.allclose(weights.sum(1), 1.) else None)
        for slot in range(int(data["slots"])):
            self.parts.append({
                "points": rotated(slot, "points"), "normals": rotated(slot, "normals"),
                "joints": arr(slot, "joints", wp.int32), "weights": arr(slot, "weights", wp.float32),
                "indices": arr(slot, "indices", wp.int32), "uv": arr(slot, "uv", wp.vec2),
                "outputs": [None]*num_agents,
            })
        from omega_playground.viewer.equipment import asset_path, load_equipment

        self.equipment_path = equipment_path
        if armed and not soma and self.equipment_path is None:
            self.equipment_path = asset_path(self.adapter.grips, native=self.native)

        self.props = [{**p, "index": body_names.index(p["body"]),
                       "vertices": np.asarray(p["vertices"], np.float64),
                       "normals": np.asarray(p["normals"], np.float64),
                       "indices": wp.array(np.asarray(p["faces"], np.int32).reshape(-1), dtype=wp.int32,
                                           device=device)}
                      for p in ([] if soma or not armed else load_equipment(self.adapter.grips, native=self.native,
                                                                           prepared_path=self.equipment_path))]
        self.hand_equipment = None
        if soma and armed:
            from omega_playground.viewer.hand_equipment import HandEquipment
            self.hand_equipment = HandEquipment(body_names, num_agents, device, self.adapter, prepared_path=equipment_path)
            self.equipment_path = self.hand_equipment.prepared_path
        self.rigid_skin = self.native and all(b is not None for b in self.rigid_bones)
        for prop in self.props:
            prop["points_gpu"] = wp.array(prop["vertices"].astype(np.float32), dtype=wp.vec3, device=device)
            prop["normals_gpu"] = wp.array(prop["normals"].astype(np.float32), dtype=wp.vec3, device=device)
        # Deforming avatars share one contiguous output allocation. Read it back
        # once per attribute, not once per mesh/robot; topology stays on the renderer.
        total = sum(len(p["points"]) for p in self.parts)*num_agents
        self.batch_points = wp.empty(total, dtype=wp.vec3, device=device)
        self.batch_normals = wp.empty_like(self.batch_points)
        self.batch_slices = []
        offset = 0
        for agent in range(num_agents):
            for slot, part in enumerate(self.parts):
                end = offset+len(part["points"])
                part["outputs"][agent] = (self.batch_points[offset:end], self.batch_normals[offset:end])
                self.batch_slices.append((f"skin_{agent:03d}_{slot}", offset, end))
                offset = end
        self.frames = 0

    def _pose_matrices(self, q):
        matrices = self.adapter.matrices(q[:, :3], q[:, 3:])
        if self.sole is not None:
            dz = self.sole.physical_height(q[:, :3], q[:, 3:])-self.sole.visual_height(matrices)
            matrices[:, 2, 3] += dz
            q = q.copy()  # Props follow the same visual shift; never alter physics.
            q[:, 2] += dz
        return q, matrices.astype(np.float32)

    def _log_props(self, viewer, agent: int, q: np.ndarray, matrices: np.ndarray) -> None:
        for prop in self.props:
            mount = self.adapter.equipment_transform(prop["body"], matrices) if self.native else None
            if mount is None:
                pose = q[prop["index"]]
                rot, origin = Rotation.from_quat(pose[3:]).as_matrix(), pose[:3]
            else:
                rot, origin = mount[:3, :3], mount[:3, 3]
            points = (prop["vertices"] @ rot.T + origin).astype(np.float32)
            normals = (prop["normals"] @ rot.T).astype(np.float32)
            viewer.log_mesh(f"prop_{agent:03d}_{prop['name']}", wp.array(points, dtype=wp.vec3, device=self.device),
                            prop["indices"], normals=wp.array(normals, dtype=wp.vec3, device=self.device),
                            backface_culling=False, color=prop["color"], roughness=prop["roughness"],
                            metallic=prop["metallic"])

    def log(self, viewer, body_q: np.ndarray) -> None:
        if len(body_q) != self.num_agents * self.bodies:
            raise ValueError(f"body_q has {len(body_q)} bodies, expected {self.num_agents} x {self.bodies}")
        # Opt-in until actual image and throughput comparison gates are complete.
        import os
        if (self.frames and self.skin.meta.get('kind') == 'soma23'
                and getattr(viewer, 'renderer', None) is not None
                and os.environ.get('OMEGA_SOMA_GPU_SKIN') == '1'):
            if not hasattr(self, '_tensor_updater'):
                from .soma_skin_tensor import TensorSkinUpdater
                self._tensor_updater = TensorSkinUpdater(self, viewer)
            self._tensor_updater.log(body_q)
            self.frames += 1
            return
        if self.hand_equipment is not None:
            self.hand_equipment.log(viewer, body_q)
        if self.optimized and hasattr(viewer, 'update_rigid_meshes'):
            self._log_optimized(viewer, body_q)
            self.frames += 1
            return
        for agent in range(self.num_agents):
            q = body_q[agent * self.bodies:(agent + 1) * self.bodies]
            q, matrices = self._pose_matrices(q)
            if not np.isfinite(matrices).all():
                raise FloatingPointError("nonfinite skin palette")
            self.palettes[agent].assign(matrices)
            self._log_props(viewer, agent, q, matrices.astype(np.float64))
            for slot, part in enumerate(self.parts):
                points, normals = part["outputs"][agent]
                wp.launch(_skin, dim=len(part["points"]),
                          inputs=[part["points"], part["normals"], part["joints"], part["weights"],
                                  self.palettes[agent], part["weights"].shape[1]],
                          outputs=[points, normals], device=self.device)
                name = f"skin_{agent:03d}_{slot}"
                # Newton's RTX viewer drops the texture when given None; pass it on every update.
                viewer.log_mesh(name, points, part["indices"], normals=normals, uvs=part["uv"],
                                texture=str(self.skin.root / f"basecolor_{slot}.png"), color=(1.0, 1.0, 1.0),
                                roughness=0.45, metallic=0.2, backface_culling=not self.native)
                if self.frames == 0:
                    self._bind_pbr(viewer, name, slot)
        self.frames += 1

    def _log_optimized(self, viewer, body_q):
        names, poses = [], []
        for agent in range(self.num_agents):
            q = body_q[agent*self.bodies:(agent+1)*self.bodies]
            q, matrices = self._pose_matrices(q)
            if not np.isfinite(matrices).all():
                raise FloatingPointError("nonfinite skin palette")
            mounts = {}
            for prop in self.props:
                body = prop["body"]
                if body not in mounts:
                    mount = self.adapter.equipment_transform(body, matrices) if self.native else None
                    mounts[body] = (np.r_[mount[:3, 3], Rotation.from_matrix(mount[:3, :3]).as_quat()]
                                    if mount is not None else q[prop["index"]])
                name = f"prop_{agent:03d}_{prop['name']}"
                names.append(name)
                poses.append(mounts[body])
                if self.frames == 0:
                    viewer.log_mesh(name, prop["points_gpu"], prop["indices"], normals=prop["normals_gpu"],
                                    backface_culling=False, color=prop["color"], roughness=prop["roughness"],
                                    metallic=prop["metallic"])
            if not self.rigid_skin:
                self.palettes[agent].assign(matrices)
            else:
                rigid = matrices[self.rigid_bones]
                rigid_poses = np.column_stack((rigid[:, :3, 3], Rotation.from_matrix(rigid[:, :3, :3]).as_quat()))
            for slot, part in enumerate(self.parts):
                name = f"skin_{agent:03d}_{slot}"
                if self.rigid_skin:
                    names.append(name)
                    poses.append(rigid_poses[slot])
                    points, normals = part["points"], part["normals"]
                else:
                    points, normals = part["outputs"][agent]
                    wp.launch(_skin, dim=len(part["points"]),
                              inputs=[part["points"], part["normals"], part["joints"], part["weights"],
                                      self.palettes[agent], part["weights"].shape[1]],
                              outputs=[points, normals], device=self.device)
                if self.frames == 0:
                    viewer.log_mesh(name, points, part["indices"], normals=normals, uvs=part["uv"],
                                    texture=str(self.skin.root/f"basecolor_{slot}.png"), color=(1., 1., 1.),
                                    roughness=.45, metallic=.2, backface_culling=not self.native)
                    self._bind_pbr(viewer, name, slot)
        if names:
            viewer.update_rigid_meshes(names, np.asarray(poses, np.float32))
        if not self.rigid_skin and self.frames:
            viewer.update_deformed_meshes(self.batch_slices, self.batch_points, self.batch_normals)

    def _bind_pbr(self, viewer, mesh_name: str, slot: int) -> None:
        if hasattr(viewer, 'bind_skin_material'):
            viewer.bind_skin_material(mesh_name, self.skin.root, slot)
            return
        from pxr import Gf, Sdf, UsdShade

        prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify(mesh_name)))
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0] if prim else None
        if not material:
            raise RuntimeError("skin meshes must be logged before RTX stage initialization")
        if self.native:
            # Thin rigid robot covers are not watertight or consistently wound.
            from pxr import UsdGeom

            UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
        path = str(material.GetPath())
        surface = UsdShade.Shader.Get(viewer.stage, path + "/PreviewSurface")
        st = UsdShade.Shader.Get(viewer.stage, path + "/PrimvarSt")
        for channel in TEXTURE_CHANNELS[1:]:
            if not (self.skin.root / f"{channel}_{slot}.png").exists():
                continue
            node = UsdShade.Shader.Define(viewer.stage, path + "/" + channel)
            node.CreateIdAttr("UsdUVTexture")
            node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(self.skin.root / f"{channel}_{slot}.png"))
            node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            if channel == "normal":
                node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                node.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                surface.CreateInput(channel, Sdf.ValueTypeNames.Normal3f).ConnectToSource(node.ConnectableAPI(), "rgb")
            else:
                node.CreateOutput("r", Sdf.ValueTypeNames.Float)
                surface.CreateInput(channel, Sdf.ValueTypeNames.Float).ConnectToSource(node.ConnectableAPI(), "r")
