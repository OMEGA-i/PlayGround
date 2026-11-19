"""Offline native G1/A3 Armed geometry and pose-only equipment animation."""
import hashlib

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .equipment import asset_path, load_equipment
from .hand_equipment import HandMounts
from .native_grip_assets import asset_path as hand_asset_path


def add_equipment(spec, package, count):
    a3 = package.robot['id'].startswith('agibot_a3')
    if not a3 and package.robot['id'] != 'unitree_g1_29dof_with_hand':
        raise ValueError('MJWarp Armed preparation requires native G1 or A3')
    path = hand_asset_path(package)
    with np.load(path) as data:
        centers = {side: data[side+'_center'].tolist() for side in ('l', 'r')}
        if a3:
            for agent in range(count):
                prefix = '' if agent == 0 else f'agent_{agent:03d}_'
                for side, word in [('l', 'left'), ('r', 'right')]:
                    name = f'omega_grip_{agent:03d}_{side}'
                    spec.add_mesh(name=name, uservert=data[side+'_points'].reshape(-1),
                                  userface=data[side+'_indices'].reshape(-1),
                                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
                    candidates = [g for b in spec.bodies for g in b.geoms
                                  if g.meshname == prefix+word+'_hand_Link' and g.contype == 0]
                    if len(candidates) != 1:
                        raise ValueError('Expected one authored A3 hand visual per side')
                    geom = candidates[0]
                    geom.meshname = name
                    geom.pos = [0., 0., 0.]
                    geom.quat = [1., 0., 0., 0.]
    mounts = add_prop_meshes(spec,count)
    return {**mounts, 'kind': 'native_a3' if a3 else 'native_g1', 'centers': centers,
            'hands_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def add_prop_meshes(spec,count):
    grips = {'l': (np.zeros(3), np.array([1., 0., 0.])),
             'r': (np.zeros(3), np.array([0., 0., 1.]))}
    parts = load_equipment(grips)
    body_names = []
    for agent in range(count):
        bodies = {}
        for role in ('shield', 'right_hand'):
            name = f'omega_equipment_{agent:03d}_{role}'
            bodies[role] = spec.worldbody.add_body(name=name)
            body_names.append(name)
        for j, part in enumerate(parts):
            name = f'omega_equipment_{agent:03d}_{j}'
            spec.add_mesh(name=name, uservert=np.asarray(part['vertices']).reshape(-1),
                          userface=np.asarray(part['faces']).reshape(-1),
                          inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
            spec.add_material(name=name, rgba=[*part['color'][:3], 1.],
                              metallic=part['metallic'], roughness=part['roughness'])
            bodies[part['body']].add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                                          meshname=name, material=name, contype=0, conaffinity=0)
    return {'body_names': body_names,'equipment_sha256': hashlib.sha256(asset_path(grips).read_bytes()).hexdigest()}


def apply_prepared_hands(model, package, count):
    """Bake authored finger geometry onto its wrist; no runtime hand actuation."""
    if package.robot['id'].startswith('agibot_a3'):
        return  # Prepared mesh was installed in the MjSpec before compilation.
    with np.load(hand_asset_path(package)) as data:
        for agent in range(count):
            prefix = '' if agent == 0 else f'agent_{agent:03d}_'
            for side, word in [('l', 'left'), ('r', 'right')]:
                wrist = model.body(prefix+word+'_wrist_yaw_link').id
                for name, transform in zip(data[side+'_names'].tolist(), data[side+'_transforms']):
                    body = model.body(prefix+name).id
                    for geom in np.flatnonzero(model.geom_bodyid == body):
                        rotation = Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).as_matrix()
                        model.geom_pos[geom] = transform[:3, 3]+transform[:3, :3]@model.geom_pos[geom]
                        q = Rotation.from_matrix(transform[:3, :3]@rotation).as_quat()
                        model.geom_quat[geom] = q[[3, 0, 1, 2]]
                        model.geom_bodyid[geom] = wrist


class ArmedPoseAdapter:
    def __init__(self, meta):
        self.names = meta['policy_body_names']
        self.count = meta['num_agents']
        self.mounts = HandMounts(self.names, self.count, 'cpu', centers=meta['armed']['centers'], sockets=meta['armed'].get('sockets'))

    def extend(self, poses):
        poses = np.asarray(poses, dtype=np.float32)
        if poses.shape != (self.count*len(self.names), 7) or not np.isfinite(poses).all():
            raise ValueError('Armed renderer requires the authoritative physical body poses')
        result = []
        for agent in range(self.count):
            mounts = self.mounts.mounts(poses[agent*len(self.names):(agent+1)*len(self.names)])
            for role in ('shield', 'right_hand'):
                m = mounts[role]
                result.append(np.r_[m[:3, 3], Rotation.from_matrix(m[:3, :3]).as_quat()])
        return np.concatenate((poses, np.asarray(result, dtype=np.float32)))
