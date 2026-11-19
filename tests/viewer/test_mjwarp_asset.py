import hashlib
import json
from types import SimpleNamespace

import mujoco
import pytest

from omega_playground.viewer.mjwarp_asset import load_visual_model


def test_prepared_visual_identity_and_integrity(tmp_path):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name='root').add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.1, 0, 0])
    mujoco.mj_saveModel(spec.compile(), str(tmp_path/'visual.mjb'))
    meta = {'schema': 'omega.mjwarp-visual/1', 'scene_sha256': 'scene', 'policy_sha256': 'policy',
            'body_names': ['root'], 'mujoco_version': mujoco.__version__, 'entry': 'visual.mjb'}
    (tmp_path/'asset.json').write_text(json.dumps(meta))
    sums = {name: hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in ('asset.json', 'visual.mjb')}
    (tmp_path/'SHA256SUMS.json').write_text(json.dumps(sums))
    package = SimpleNamespace(body_names=['root'], sha256sum=lambda: 'policy')
    model, _ = load_visual_model(tmp_path, package, 'scene')
    assert model.ngeom == 1
    with pytest.raises(ValueError, match='does not match'):
        load_visual_model(tmp_path, package, 'different-scene')
    (tmp_path/'visual.mjb').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_visual_model(tmp_path, package, 'scene')


def test_three_agent_armed_bundle_contract(tmp_path):
    spec = mujoco.MjSpec()
    bodies = ['root', 'agent_001_root', 'agent_002_root']
    props = [f'omega_equipment_{i:03d}_{role}' for i in range(3) for role in ('shield', 'right_hand')]
    for name in bodies+props:
        spec.worldbody.add_body(name=name).add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.01, 0, 0])
    mujoco.mj_saveModel(spec.compile(), str(tmp_path/'visual.mjb'))
    meta = {'schema': 'omega.mjwarp-visual/1', 'scene_sha256': 'scene', 'policy_sha256': 'policy',
            'body_names': bodies+props, 'policy_body_names': ['root'], 'num_agents': 3,
            'mujoco_version': mujoco.__version__, 'entry': 'visual.mjb',
            'armed': {'kind': 'native_g1', 'body_names': props, 'centers': {'l': [.1, 0., 0.], 'r': [.1, 0., 0.]}}}
    def save():
        (tmp_path/'asset.json').write_text(json.dumps(meta))
        (tmp_path/'SHA256SUMS.json').write_text(json.dumps({
            name: hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in ('asset.json', 'visual.mjb')}))
    save()
    package = SimpleNamespace(body_names=['root'], robot={'id': 'unitree_g1_29dof_with_hand'}, sha256sum=lambda: 'policy')
    load_visual_model(tmp_path, package, 'scene', 3, armed=True)
    with pytest.raises(ValueError, match='selection mismatch'):
        load_visual_model(tmp_path, package, 'scene', 3, armed=False)
    with pytest.raises(ValueError, match='contract mismatch|order mismatch'):
        load_visual_model(tmp_path, package, 'scene', 2)
    meta['armed']['kind'] = 'native_a3'
    save()
    with pytest.raises(ValueError, match='identity mismatch'):
        load_visual_model(tmp_path, package, 'scene', 3)
    meta['armed']['kind'] = 'native_g1'
    meta['armed']['centers']['l'][0] = float('nan')
    save()
    with pytest.raises(ValueError, match='finite XYZ'):
        load_visual_model(tmp_path, package, 'scene', 3)
