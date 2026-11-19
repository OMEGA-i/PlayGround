"""Load the offline visual artifact without importing USD or running conversion."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def load_visual_model(directory, package, scene_sha256, count=1, *, armed=None):
    if type(count) is not int or not 1 <= count <= 9:
        raise ValueError('Visual agent count must be between one and nine')
    root = Path(directory).resolve()
    sums = json.loads((root/'SHA256SUMS.json').read_text())
    if 'asset.json' not in sums:
        raise ValueError('Visual metadata is not checksummed')
    for name, expected in sums.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Visual asset path missing or outside its bundle')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Visual asset checksum mismatch: {name}')
    meta = json.loads((root/'asset.json').read_text())
    if meta.get('schema') != 'omega.mjwarp-visual/1':
        raise ValueError('Unsupported visual asset schema')
    if meta['scene_sha256'] != scene_sha256 or meta['policy_sha256'] != package.sha256sum():
        raise ValueError('Prepared visual asset does not match scene/policy')
    expected = [name if i == 0 else f'agent_{i:03d}_'+name for i in range(count) for name in package.body_names]
    if armed is not None and bool(meta.get('armed')) != armed:
        raise ValueError('Prepared visual Armed selection mismatch')
    if meta.get('armed'):
        robot = package.robot['id']
        expected_kind = ('native_a3' if robot.startswith('agibot_a3') else
                         'native_g1' if robot == 'unitree_g1_29dof_with_hand' else None)
        if meta['armed']['kind'] != expected_kind or meta.get('policy_body_names') != package.body_names:
            raise ValueError('Prepared Armed robot identity mismatch')
        centers = meta['armed'].get('centers', {})
        if set(centers) != {'l', 'r'} or any(np.asarray(v).shape != (3,) or not np.isfinite(v).all() for v in centers.values()):
            raise ValueError('Prepared Armed palm centers must be finite XYZ vectors')
        expected.extend(f'omega_equipment_{i:03d}_{role}' for i in range(count) for role in ('shield', 'right_hand'))
        if meta['armed']['kind'] not in ('native_g1', 'native_a3') or meta['armed']['body_names'] != expected[count*len(package.body_names):]:
            raise ValueError('Prepared Armed body contract mismatch')
    if meta.get('num_agents', 1) != count or meta['body_names'] != expected:
        raise ValueError('Prepared visual body order mismatch')
    if meta['mujoco_version'] != mujoco.__version__:
        raise ValueError('Reprepare the binary visual asset with the runtime MuJoCo version')
    if meta['entry'] not in sums:
        raise ValueError('Visual model is not checksummed')
    model = mujoco.MjModel.from_binary_path(str(root/meta['entry']))
    actual = {model.body(i).name for i in range(1, model.nbody)}
    if not set(expected).issubset(actual):
        raise ValueError('Compiled visual model is missing declared pose bodies')
    return model, meta
