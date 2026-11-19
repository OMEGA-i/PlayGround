import hashlib
import json
from types import SimpleNamespace

import pytest

from omega_playground.ops.prepared_assets import PreparedAssetRegistry


def bundle(root, meta):
    root.mkdir(exist_ok=True)
    (root/'model.bin').write_bytes(b'compiled fixture')
    (root/'asset.json').write_text(json.dumps({'entry': 'model.bin', **meta}))
    (root/'SHA256SUMS.json').write_text(json.dumps({
        name: hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('asset.json', 'model.bin')}))


@pytest.fixture
def prepared(tmp_path):
    package = SimpleNamespace(sha256sum=lambda: 'policy', family='sonic', robot={'id': 'unitree_g1'})
    bundle(tmp_path/'physics', {'policy_sha256': 'policy', 'joint_physics': {'hip': {}}})
    bundle(tmp_path/'visual', {'schema': 'omega.mjwarp-visual/1', 'policy_sha256': 'policy',
                             'scene_sha256': 'scene', 'num_agents': 3, 'armed': False})
    entry = {'policy_sha256': 'policy', 'scene_sha256': 'scene', 'avatar': 'g1', 'armed': False,
             'num_agents': 3, 'physics_asset': 'physics', 'visual_model': 'visual'}
    path = tmp_path/'registry.json'
    path.write_text(json.dumps({'schema': 'omega.prepared-runtime/1', 'entries': [entry]}))
    return path, package, entry


def test_exact_selection_and_revalidation(prepared):
    path, package, _ = prepared
    registry = PreparedAssetRegistry(path)
    selected = registry.select(package, 'scene', 'g1', False, 3)
    assert selected['physics_asset'] == str(path.parent/'physics')
    (path.parent/'visual/model.bin').write_bytes(b'corrupted after previous validation')
    with pytest.raises(ValueError, match='checksum'):
        registry.select(package, 'scene', 'g1', False, 3)


@pytest.mark.parametrize('scene,avatar,armed,count', [
    ('other', 'g1', False, 3), ('scene', 'a3', False, 3),
    ('scene', 'g1', True, 3), ('scene', 'g1', False, 9)])
def test_missing_combination_is_not_substituted(prepared, scene, avatar, armed, count):
    path, package, _ = prepared
    with pytest.raises(ValueError, match='离线资产'):
        PreparedAssetRegistry(path).select(package, scene, avatar, armed, count)


@pytest.mark.parametrize('field,value', [('scene_sha256', 'wrong'), ('policy_sha256', 'wrong'),
                                        ('num_agents', 9), ('armed', True)])
def test_checksummed_but_incompatible_visual_rejected(prepared, field, value):
    path, package, _ = prepared
    root = path.parent/'visual'
    meta = json.loads((root/'asset.json').read_text())
    bundle(root, {**meta, field: value})
    with pytest.raises(ValueError, match='selection mismatch'):
        PreparedAssetRegistry(path).select(package, 'scene', 'g1', False, 3)


def test_duplicate_selector_rejected(prepared):
    path, _, entry = prepared
    path.write_text(json.dumps({'schema': 'omega.prepared-runtime/1', 'entries': [entry, entry]}))
    with pytest.raises(ValueError, match='Duplicate'):
        PreparedAssetRegistry(path)


def test_rebuild_keeps_registry():
    from omega_playground.ops.robot import restart_arguments
    args = SimpleNamespace(package='p', scene='s', map='m', device='cpu', port=1, render_hz=None,
                           render_width=640, render_height=480, camera='qc', render_profile='realtime',
                           prepared_assets='/registry.json', render_environment='/environment')
    result = restart_arguments(args, 'player', 'output')
    assert result[result.index('--prepared-assets') + 1] == '/registry.json'
    assert result[result.index('--render-environment') + 1] == '/environment'


def test_optional_environment_is_scene_checked_and_absence_clears_selection(prepared):
    path, package, entry = prepared
    env = path.parent/'environment'
    bundle(env, {'schema':'omega.mjwarp-environment/1', 'scene_sha256':'scene'})
    entry['render_environment'] = 'environment'
    path.write_text(json.dumps({'schema':'omega.prepared-runtime/1', 'entries':[entry]}))
    selected = PreparedAssetRegistry(path).select(package, 'scene', 'g1', False, 3)
    assert selected['render_environment'] == str(env)
    bundle(env, {'schema':'omega.mjwarp-environment/1', 'scene_sha256':'wrong'})
    with pytest.raises(ValueError, match='environment scene/schema'):
        PreparedAssetRegistry(path).select(package, 'scene', 'g1', False, 3)
    del entry['render_environment']
    path.write_text(json.dumps({'schema':'omega.prepared-runtime/1', 'entries':[entry]}))
    assert PreparedAssetRegistry(path).select(package, 'scene', 'g1', False, 3)['render_environment'] is None
