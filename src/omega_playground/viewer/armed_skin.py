"""Offline grip preparation in the avatar's authored bind coordinates."""
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from omega_playground.viewer.native_skin import native_grip, prepare_grip_mesh


def prepare_soma_grip(root: Path):
    source = root / 'skin.npz'
    original = np.load(source)
    data = dict(original)
    names, bind, parents = data['names'].tolist(), data['bind'], data['parents']
    native = 'left_hand' in names
    mounts = []
    if native:
        data = prepare_grip_mesh(data, root.name)
    rotations = {}
    finger_owner = {}
    for side, sign in [('l', -1), ('r', 1)]:
        hand = names.index(('left' if side == 'l' else 'right') + '_hand' if native else f'hand_{side}')
        if native:
            center = bind[hand, :3, 3] + bind[hand, :3, :3] @ native_grip(root.name, side)
            frame = bind[hand, :3, :3]
        else:
            knuckles = np.array([bind[names.index(f'{f}_01_{side}'), :3, 3]
                                 for f in ('index', 'middle', 'ring', 'pinky')])
            axis = knuckles[0] - knuckles[-1]
            axis /= np.linalg.norm(axis)
            distal = knuckles.mean(0) - bind[hand, :3, 3]
            distal -= axis * (distal @ axis)
            distal /= np.linalg.norm(distal)
            normal = np.cross(axis, distal)
            center = knuckles.mean(0) + .025 * sign * normal + .007 * distal
            frame = np.column_stack((distal, normal, axis))
            curls = [(f'{f}_{j:02d}_{side}', angle) for f in ('index', 'middle', 'ring', 'pinky')
                     for j, angle in ((1, 48), (2, 78), (3, 55))]
            curls += [(f'thumb_{j:02d}_{side}', angle) for j, angle in ((1, -15), (2, 20), (3, 40))]
            for name, angle in curls:
                i = names.index(name)
                rotations[i] = Rotation.from_rotvec(bind[i, :3, :3].T @ axis * np.deg2rad(sign * angle)).as_matrix()
            for i, name in enumerate(names):
                if name.endswith('_'+side) and name.startswith(('thumb_', 'index_', 'middle_', 'ring_', 'pinky_')):
                    finger_owner[i] = hand
        mount = np.eye(4)
        mount[:3, :3], mount[:3, 3] = frame, center
        mounts.append(mount)
    if not native:
        target = bind.copy()
        inverse = np.linalg.inv(bind)
        for i, parent in enumerate(parents):
            if parent >= 0:
                target[i] = target[parent] @ inverse[parent] @ bind[i]
            if i in rotations:
                target[i, :3, :3] = target[i, :3, :3] @ rotations[i]
        matrices = target @ inverse
        for slot in range(int(data['slots'])):
            joints, weights = data[f's{slot}_joints'], data[f's{slot}_weights']
            transform = (matrices[joints] * weights[:, :, None, None]).sum(axis=1)
            points = data[f's{slot}_points']
            data[f's{slot}_points'] = np.einsum('nij,nj->ni', transform[:, :3, :3], points) + transform[:, :3, 3]
            normals = np.einsum('nij,nj->ni', transform[:, :3, :3], data[f's{slot}_normals'])
            data[f's{slot}_normals'] = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            mapped = joints.copy()
            for finger, hand in finger_owner.items():
                mapped[joints == finger] = hand
            data[f's{slot}_joints'] = mapped
    data['armed_mounts'] = np.array(mounts)
    data['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    np.savez_compressed(root / 'armed_skin.npz', **data)


def load_soma_grip(root):
    source, prepared = root / 'skin.npz', root / 'armed_skin.npz'
    if not prepared.is_file():
        raise ValueError('Missing SOMA Armed skin; run tools/assets/prepare_armed_assets.py')
    data = np.load(prepared)
    if str(data['source_sha256']) != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Stale SOMA Armed skin; run tools/assets/prepare_armed_assets.py')
    return data
