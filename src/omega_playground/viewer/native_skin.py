"""Render-only native A3 / H2 robot appearances on the ASE skeleton.

Ported from ProtoMotions-ASE-newton ``utils/ase_native_fk.py`` and ``utils/ase_native_assets.py``. The display rig
keeps the robot's authored hierarchy and bone lengths: ASE body rotations drive it and only the pelvis position is
taken from physics, so these visuals do not certify collision alignment. Hands are oriented like UE5 Manny's grip,
and the rigid finger islands are posed around the prop handles at load time.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Finger mesh islands per robot and side: (first slot of each finger, thumb slots).
FINGER_SLOTS = {
    "a3": {"l": ((24, 29, 34, 40), (19, 20, 21, 22, 23)), "r": ((60, 65, 70, 76), (55, 56, 57, 58, 59))},
    "h2": {"l": ((32, 37, 42, 48), (27, 28, 29, 30, 31)), "r": ((67, 72, 77, 83), (62, 63, 64, 65, 66))},
}
# Measured from MannyPalette's knuckle row in the ASE hand-body frame.
MANNY_GRIP_AXIS = {"r": np.array([0.88174738, 0.42017948, -0.21440791]),
                   "l": np.array([0.88174602, -0.42018471, -0.21440326])}
MANNY_DISTAL_AXIS = {"r": np.array([-0.18281599, -0.11461679, -0.97644319]),
                     "l": np.array([-0.18282499, 0.11458716, -0.97644498])}
MANNY_GRIP_CENTER = {"r": np.array([-0.01969845, 0.01565806, -0.09646503]),
                     "l": np.array([-0.01969941, -0.01566097, -0.09646320])}
HAND_DISTAL_SCALE = {"a3": 0.52, "h2": 0.45}
# Per-finger hinge angles (deg) fit to the 21.8 mm baton grip and 18 mm shield handle.
FINGER_CURL = {
    "a3": {"l": [[17.21, 39.12, 20.0], [13.61, 47.94, 20.0], [1.12, 56.42, 20.0], [-9.76, 60.24, 20.0]],
           "r": [[7.11, 48.18, 20.0], [1.21, 60.32, 20.0], [-12.65, 71.47, 20.0], [-25.09, 78.79, 20.0]]},
    "h2": {"l": [[0.46, 56.53, 20.0], [3.93, 55.27, 20.0], [-0.75, 54.11, 20.0], [-4.36, 50.96, 20.0]],
           "r": [[-16.05, 82.23, 20.0], [-12.1, 79.47, 20.0], [-16.97, 78.93, 20.0], [-20.43, 75.3, 20.0]]},
}
BASIS = Rotation.from_euler("z", -90, degrees=True).as_matrix()
ROLES = {"pelvis": "Hips", "torso": "Chest", "head": "Head"}
for _word, _side in (("left", "Left"), ("right", "Right")):
    for _role, _body in (("upper_arm", "Arm"), ("lower_arm", "ForeArm"), ("hand", "Hand"),
                         ("thigh", "Leg"), ("shin", "Shin"), ("foot", "Foot")):
        ROLES[f"{_word}_{_role}"] = _side + _body


def native_grip(robot: str, side: str) -> np.ndarray:
    """Handle center in the displayed hand frame: in the palm, just beyond the knuckle row."""
    return np.array([0.080 if robot == "a3" else 0.082, 0.030 if side == "r" else -0.030, 0.0])


def _between(a, b) -> np.ndarray:
    return Rotation.align_vectors(np.asarray([b]), np.asarray([a]))[0].as_matrix()


def _align(a, b) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    angle = np.arccos(np.clip(a @ b, -1, 1))
    if np.linalg.norm(axis) < 1e-8:
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()


def prepare_grip_mesh(data, robot: str) -> dict:
    """Pose the rigid finger islands around the prop handles, in hand space."""
    out = dict(data)
    names = data["names"].tolist()
    for side, (finger_starts, thumb_slots) in FINGER_SLOTS[robot].items():
        sign = 1.0 if side == "r" else -1.0
        bind = data["bind"][names.index(("right" if side == "r" else "left") + "_hand")]
        hand = names.index(("right" if side == "r" else "left") + "_hand")
        points, normals = {}, {}
        for slot in range(int(data["slots"])):
            influence = (data[f"s{slot}_weights"] * (data[f"s{slot}_joints"] == hand)).sum(1)
            if not np.all(influence > 0.9):
                continue
            points[slot] = (data[f"s{slot}_points"] - bind[:3, 3]) @ bind[:3, :3]
            points[slot][:, 0] *= HAND_DISTAL_SCALE[robot]
            normals[slot] = data[f"s{slot}_normals"] @ bind[:3, :3]
            normals[slot][:, 0] /= HAND_DISTAL_SCALE[robot]
            normals[slot] /= np.maximum(np.linalg.norm(normals[slot], axis=1, keepdims=True), 1e-12)

        def turn(slots, pivot, rotation, points=points, normals=normals):
            for slot in slots:
                points[slot] = (points[slot] - pivot) @ rotation.T + pivot
                normals[slot] = normals[slot] @ rotation.T

        # Bend in the scaled hand frame so segment lengths and handle radius are in metres.
        for first, angles in zip(finger_starts, FINGER_CURL[robot][side]):
            pivot0 = points[first].mean(0)
            pivot0[0] = points[first][:, 0].min()
            pivot1 = points[first + 1].mean(0)
            pivot1[0] = points[first + 1][:, 0].max()
            pivot2 = points[first + 2].mean(0)
            pivot2[0] = points[first + 2][:, 0].max()
            turn(range(first + 3, first + 5), pivot2, Rotation.from_euler("z", sign * angles[2], degrees=True).as_matrix())
            turn(range(first + 2, first + 5), pivot1, Rotation.from_euler("z", sign * angles[1], degrees=True).as_matrix())
            turn(range(first, first + 5), pivot0, Rotation.from_euler("z", sign * angles[0], degrees=True).as_matrix())

        # Two-link thumb opposition: keep the palm socket and segment lengths, bring the pad around the handle.
        base = points[thumb_slots[0]]
        root = np.array([base[:, 0].min() + 0.004, np.median(base[:, 1]), base[:, 2].min() + 0.004])
        joint = points[thumb_slots[1]].mean(0)
        tip = points[thumb_slots[-1]].mean(0)
        pad_offset = (0.0305 if robot == "a3" else 0.026) - (0.005 if side == "l" else 0.0)
        target = native_grip(robot, side) + [0.0, sign * pad_offset, 0.020]
        l1, l2 = np.linalg.norm(joint - root), np.linalg.norm(tip - joint)
        direction = target - root
        distance = np.linalg.norm(direction)
        direction /= distance
        distance = np.clip(distance, abs(l1 - l2) + 1e-6, l1 + l2 - 1e-6)
        target = root + direction * distance
        along = (l1 * l1 - l2 * l2 + distance * distance) / (2 * distance)
        pole = np.array([-1.0, sign, 0.2])
        pole -= direction * (pole @ direction)
        pole /= np.linalg.norm(pole)
        elbow = root + along * direction + np.sqrt(max(0.0, l1 * l1 - along * along)) * pole
        rotation = _between(joint - root, elbow - root)
        turn(thumb_slots, root, rotation)
        moved_tip = rotation @ (tip - root) + root
        turn(thumb_slots[2:], elbow, _between(moved_tip - elbow, target - elbow))

        for slot, slot_points in points.items():
            out[f"s{slot}_points"] = slot_points @ bind[:3, :3].T + bind[:3, 3]
            out[f"s{slot}_normals"] = normals[slot] @ bind[:3, :3].T
    return out


class Soma23Palette:
    """Authored-bind display rig driven by SOMA23-named rotations; positions only drive the root."""

    def __init__(self, data, body_names):
        self.names = data["names"].tolist()
        self.bind = data["bind"].astype(float)
        self.inverse = np.linalg.inv(self.bind)
        self.parents = data["parents"].astype(int)
        self.soma_names = list(data["soma_names"])
        self.rest = dict(zip(self.soma_names, data["soma_rest"]))
        self.indices = [body_names.index(n) for n in self.soma_names]
        self.mapping = {bone: body for body, bone in zip(self.soma_names, list(data["mapping"])) if bone}
        self.calibration = self.bind[:, :3, :3].copy()
        self.root = self.names.index("pelvis")
        ends = {}
        for side in ("Left", "Right"):
            ends.update({side + "Arm": side + "ForeArm", side + "ForeArm": side + "Hand",
                         side + "Leg": side + "Shin", side + "Shin": side + "Foot"})
        reverse = {v: k for k, v in self.mapping.items()}
        deltas = {}
        for bone, body in self.mapping.items():
            i = self.names.index(bone)
            # Native joint offsets need not lie on a straight limb axis; keep leg frames as authored.
            native_leg = bool(data.get("native_leg_frames", False)) and body.endswith(("Leg", "Shin"))
            if body in ends and not native_leg:
                child = self.names.index(reverse[ends[body]])
                deltas[body] = _align(self.bind[child, :3, 3] - self.bind[i, :3, 3],
                                      self.rest[ends[body]] - self.rest[body])
                self.calibration[i] = deltas[body] @ self.bind[i, :3, :3]
        for side in ("Left", "Right"):
            i = self.names.index(reverse[side + "Hand"])
            self.calibration[i] = deltas[side + "ForeArm"] @ self.bind[i, :3, :3]
        if "semantic_frames" in data:
            for name, frame in zip(data["semantic_names"].tolist(), data["semantic_frames"]):
                sign = 1.0 if name.startswith("left") else -1.0
                distal, across = np.array([sign, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])
                target = np.column_stack([distal, across, np.cross(distal, across)])
                i = self.names.index(name)
                self.calibration[i] = target @ frame.T @ self.bind[i, :3, :3]
        for bone in self.names:
            if bone.startswith("spine_") and bone not in self.mapping:
                self.mapping[bone] = "Chest"
        if any(parent >= i for i, parent in enumerate(self.parents)):
            raise ValueError("display skeleton must be parent-first")
        self.root_offset = np.zeros(3)
        rest = self.targets(np.array([self.rest[n] for n in self.soma_names]),
                            np.tile([0.0, 0.0, 0.0, 1.0], (len(self.soma_names), 1)), ordered=True)
        # Match the mean ankle height of the calibrated rest pose without resizing.
        self.root_offset[2] = np.mean([self.rest[s + "Foot"][2] - rest[self.names.index(reverse[s + "Foot"]), 2, 3]
                                       for s in ("Left", "Right")])

    def targets(self, positions, quaternions, ordered=False):
        if not ordered:
            positions, quaternions = positions[self.indices], quaternions[self.indices]
        state = dict(zip(self.soma_names, Rotation.from_quat(quaternions).as_matrix()))
        root_pos, root_rot = positions[self.soma_names.index("Hips")], state["Hips"]
        targets = np.tile(np.eye(4), (len(self.names), 1, 1))
        for i, name in enumerate(self.names):
            parent = self.parents[i]
            if parent >= 0:
                targets[i] = targets[parent] @ self.inverse[parent] @ self.bind[i]
            else:
                targets[i, :3, :3] = root_rot @ self.bind[i, :3, :3]
                targets[i, :3, 3] = root_pos + root_rot @ (self.bind[i, :3, 3] - self.bind[self.root, :3, 3]
                                                           + self.root_offset)
            if name in self.mapping:
                targets[i, :3, :3] = state[self.mapping[name]] @ self.calibration[i]
        return targets

    def matrices(self, positions, quaternions):
        return self.targets(positions, quaternions) @ self.inverse


class NativeASEPalette:
    """Per-bone skinning matrices for a native robot appearance from ASE body poses."""

    def __init__(self, data, body_names: list[str], robot: str):
        if list(data["ase_names"]) != list(body_names):
            raise ValueError("native skin was calibrated for a different ASE body list")
        self.robot = robot
        self.names = data["names"].tolist()
        self.body_names = list(body_names)
        self.basis = BASIS.T
        rest = dict(zip(data["ase_names"].tolist(), data["ase_rest"]))
        # ASE has no neck DOF: torso drives a virtual neck.
        self.drivers = list(ROLES) + ["torso"]
        soma_names = list(ROLES.values()) + ["Neck1"]
        mapped = dict(data)
        mapped["soma_names"] = np.array(soma_names)
        mapped["soma_rest"] = np.array([rest[n] for n in self.drivers]) @ BASIS.T
        mapped["mapping"] = np.array(list(ROLES) + ["neck"])
        self.driver_indices = [self.body_names.index(n) for n in self.drivers]
        self.fk = Soma23Palette(mapped, soma_names)
        world_basis = np.eye(4)
        world_basis[:3, :3] = BASIS.T
        self.rest_bind = world_basis @ self.fk.bind
        self.rest_bind_inverse = np.linalg.inv(self.rest_bind)
        self.grips = {side: (native_grip(robot, side), np.array([0.0, 0.0, 1.0])) for side in ("l", "r")}

    def equipment_transform(self, body: str, matrices: np.ndarray) -> np.ndarray | None:
        """World transform of a prop following the displayed hand (the native wrist is not ASE's wrist)."""
        if body == "shield":
            joint = self.names.index("left_hand")
            hand = matrices[joint] @ self.rest_bind[joint]
            across, distal = MANNY_GRIP_AXIS["l"], MANNY_DISTAL_AXIS["l"]
            hand_frame = np.column_stack((distal, np.cross(across, distal), across))
            handle = np.array([0.0, -0.07, -0.138947]) + MANNY_GRIP_CENTER["l"]
            mount = np.eye(4)
            mount[:3, :3] = hand_frame.T
            mount[:3, 3] = self.grips["l"][0] - mount[:3, :3] @ handle
            return hand @ mount
        if body not in ("left_hand", "right_hand"):
            return None
        joint = self.names.index(body)
        return matrices[joint] @ self.rest_bind[joint]

    def matrices(self, positions, quaternions):
        body_rotations = Rotation.from_quat(np.asarray(quaternions)).as_matrix()
        positions = np.asarray(positions)[self.driver_indices] @ BASIS.T
        r = body_rotations[self.driver_indices]
        q = Rotation.from_matrix(BASIS @ r @ BASIS.T).as_quat()
        # The arm chain keeps its authored joint offsets; ASE body centers would pull the shoulder armor off the torso.
        m = self.fk.matrices(positions, q)
        c = np.eye(4)
        c[:3, :3] = BASIS
        m = np.linalg.inv(c) @ m @ c
        for side, body in (("r", "right_hand"), ("l", "left_hand")):
            joint = self.names.index(body)
            visual = m[joint] @ self.rest_bind[joint]
            hand_frame = np.column_stack((MANNY_DISTAL_AXIS[side],
                                          np.cross(MANNY_GRIP_AXIS[side], MANNY_DISTAL_AXIS[side]),
                                          MANNY_GRIP_AXIS[side]))
            visual[:3, :3] = body_rotations[self.body_names.index(body)] @ hand_frame
            m[joint] = visual @ self.rest_bind_inverse[joint]
        return m
