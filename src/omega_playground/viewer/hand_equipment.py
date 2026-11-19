"""Shared render-only ASE props, attached to native or skinned SONIC palms."""
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from omega_playground.viewer.equipment import load_equipment

# The common mesh has its baton grip at zero and its shield handle at this point.
SHIELD_HANDLE = np.array([0., -.07, -.138947])


class HandMounts:
    def __init__(self, body_names, num_agents, device, palette=None, centers=None, sockets=None):
        self.bodies, self.num_agents = len(body_names), num_agents
        self.device, self.palette = device, palette
        self.frames = 0
        self.hands = {}
        self.forearm = None
        if sockets:
            self.forearm=body_names.index(sockets['forearm'])
        elif palette is None:
            self.forearm = next((i for i, n in enumerate(body_names) if n.lower() == "left_elbow_link"), None)
        else:
            self.forearm = palette.names.index(next(n for n, body in palette.mapping.items() if body == "LeftForeArm"))
        for side, word in [('l', 'left'), ('r', 'right')]:
            if sockets:
                mount=sockets[side]
                self.hands[side]=(body_names.index(mount['name']),np.asarray(mount['rotation']),np.asarray(centers[side]))
            elif palette is None:
                name = next((n for n in body_names if n.lower() == word+'_wrist_yaw_link'), None)
                if name is None:
                    raise ValueError(f'No equipment mount for {word} hand')
                # Both released G1/A3 meshes extend along +X; knuckles span Z.
                self.hands[side] = (body_names.index(name), np.eye(3), np.asarray(centers[side]) if centers is not None else np.array([.09, 0., 0.]))
            else:
                role = ('Left' if side == 'l' else 'Right')+'Hand'
                name = next(n for n, body in palette.mapping.items() if body == role)
                index = palette.names.index(name)
                if palette.armed_mounts is None:
                    raise ValueError('Armed skin lacks prepared palm mounts')
                mount = palette.armed_mounts[0 if side == 'l' else 1]
                self.hands[side] = (index, mount[:3, :3], mount[:3, 3])
    def mounts(self, q):
        matrices = None if self.palette is None else self.palette.matrices(q[:, :3], q[:, 3:])
        result = {}
        for side, (index, frame, center) in self.hands.items():
            if matrices is None:
                rotation = Rotation.from_quat(q[index, 3:]).as_matrix()
                origin = q[index, :3]
            else:
                rotation, origin = matrices[index, :3, :3], matrices[index, :3, 3]
            rot = rotation @ frame
            grip = origin + rotation @ center
            if side == 'l':
                # Shield handle crosses the palm; strap runs proximally along the forearm.
                rot = rot @ np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
                if self.forearm is not None:
                    elbow = (q[self.forearm, :3] if matrices is None else
                             (matrices[self.forearm] @ self.palette.bind[self.forearm])[:3, 3])
                    # Revolute freedom about the cylindrical grip preserves palm contact.
                    # Align the shield/strap with the displayed forearm, not wrist flexion.
                    across = rot[:, 0]
                    proximal = elbow - grip
                    proximal -= across * (across @ proximal)
                    length = np.linalg.norm(proximal)
                    if length > 1e-6:
                        proximal /= length
                        normal = np.cross(proximal, across)
                        # A strongly flexed wrist can still enter the plate even when
                        # the elbow is aligned. Keep both ends of the forearm behind
                        # the shield, rotating only about the grip cylinder.
                        wrist = (q[index, :3] if matrices is None else
                                 (matrices[index] @ self.palette.bind[index])[:3, 3])
                        offsets = np.array([elbow-grip, wrist-grip])
                        limit = .07 - .055  # back-plane gap minus forearm envelope
                        if np.max(offsets @ normal) > limit:
                            angles = np.deg2rad(np.arange(-180., 181., 1.))
                            normals = np.cos(angles)[:, None]*normal + np.sin(angles)[:, None]*proximal
                            valid = np.max(offsets @ normals.T, axis=0) <= limit
                            choice = np.argmin(np.where(valid, np.abs(angles), np.inf))
                            if valid[choice]:
                                normal = normals[choice]
                                proximal = np.cross(across, normal)
                        rot = np.column_stack((across, normal, proximal))
            mount = np.eye(4)
            mount[:3, :3] = rot
            mount[:3, 3] = grip - (rot @ SHIELD_HANDLE if side == 'l' else 0.)
            result['shield' if side == 'l' else 'right_hand'] = mount
        return result


class HandEquipment(HandMounts):
    def __init__(self, body_names, num_agents, device, palette=None, centers=None, prepared_path=None):
        super().__init__(body_names, num_agents, device, palette, centers)
        grips = {'l': (np.zeros(3), np.array([1., 0., 0.])),
                 'r': (np.zeros(3), np.array([0., 0., 1.]))}
        self.parts = []
        from omega_playground.viewer.equipment import asset_path
        self.prepared_path = prepared_path if prepared_path is not None else asset_path(grips)
        for part in load_equipment(grips, prepared_path=self.prepared_path):
            self.parts.append({**part,
                'points': wp.array(np.asarray(part['vertices'], np.float32), dtype=wp.vec3, device=device),
                'normals': wp.array(np.asarray(part['normals'], np.float32), dtype=wp.vec3, device=device),
                'indices': wp.array(np.asarray(part['faces'], np.int32).reshape(-1), dtype=wp.int32, device=device)})

    def log(self, viewer, body_q):
        names, poses = [], []
        for agent in range(self.num_agents):
            mounts = self.mounts(body_q[agent*self.bodies:(agent+1)*self.bodies])
            for part in self.parts:
                name = f'hand_prop_{agent:03d}_{part["name"]}'
                mount = mounts[part['body']]
                names.append(name)
                poses.append(np.r_[mount[:3, 3], Rotation.from_matrix(mount[:3, :3]).as_quat()])
                if self.frames == 0:
                    viewer.log_mesh(name, part['points'], part['indices'], normals=part['normals'],
                                    color=part['color'], roughness=part['roughness'], metallic=part['metallic'],
                                    backface_culling=False)
        viewer.update_rigid_meshes(names, np.asarray(poses, np.float32))
        self.frames += 1
