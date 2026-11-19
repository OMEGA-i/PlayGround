from pathlib import Path

import mujoco
import numpy as np
import pytest

from omega_playground.policy.package import load
from omega_playground.viewer import equipment
from omega_playground.viewer.mjwarp_skin import SkinMeshScene
from omega_playground.viewer.mjwarp_skin_asset import load_prepared_skin, prepare_skin


@pytest.mark.parametrize('appearance', ['ue4_mannequin', 'h2'])
def test_load_prepared_skin_without_compiler_or_external_equipment(tmp_path, monkeypatch, appearance):
    root = Path(__file__).resolve().parents[2]
    policy = root/'artifacts/policies/ase-native13b-flat'
    skin = root/'artifacts/skins'/appearance
    if not policy.exists() or not skin.exists():
        pytest.skip('ASE assets not installed')
    package = load(policy)
    model = mujoco.MjModel.from_xml_path(str(package.asset_path))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    ids = [model.body(n).id for n in package.body_names]
    poses = np.c_[data.xpos[ids], data.xquat[ids][:, [1, 2, 3, 0]]]
    bundle = tmp_path/'prepared'
    meta = prepare_skin(package, skin, bundle, armed=True)

    def forbidden(*args, **kwargs):
        raise AssertionError('Loader must not compile or read external equipment')
    monkeypatch.setattr(SkinMeshScene, 'compile', forbidden)
    monkeypatch.setattr(mujoco.MjModel, 'from_xml_path', forbidden)
    monkeypatch.setattr(equipment, 'asset_path', forbidden)
    poses[:, 0] += 10  # Runtime initial pose differs from the compile pose.
    loaded, painter, bridge, actual = load_prepared_skin(bundle, package, poses, armed=True, device='cpu')
    assert actual == meta
    assert loaded.nmesh == len(bridge.meshes)
    if not painter.rigid_skin:
        assert bridge.dynamic  # First-frame outputs must already be scheduled for upload.
    for name, points in bridge.dynamic.items():
        assert points is bridge.meshes[name]['points']
    assert Path(painter.equipment_path) == bundle/'equipment.npz'
    # Render workers start before receiving the first authoritative snapshot.
    load_prepared_skin(bundle, package, armed=True, device='cpu')
    with pytest.raises(ValueError, match='scene mismatch'):
        load_prepared_skin(bundle, package, poses, armed=True, device='cpu', scene_sha256='wrong')
    with pytest.raises(ValueError, match='avatar mismatch'):
        load_prepared_skin(bundle, package, poses, armed=True, device='cpu', skin_manifest_sha256='wrong')
    with pytest.raises(ValueError, match='selection mismatch'):
        load_prepared_skin(bundle, package, poses, armed=False, device='cpu')
    path = bundle/'equipment.npz'
    path.write_bytes(path.read_bytes()+b'corrupted')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_prepared_skin(bundle, package, poses, armed=True, device='cpu')
