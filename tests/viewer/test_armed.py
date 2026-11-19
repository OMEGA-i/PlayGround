from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omega_playground.ops.robot import restart_arguments
from omega_playground.viewer.hand_equipment import SHIELD_HANDLE, HandEquipment

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('armed', [False, True])
def test_restart_preserves_explicit_equipment_choice(armed):
    args = SimpleNamespace(package='p', scene='s', map='m', device='cpu', port=1, render_hz=None,
                           render_width=640, render_height=480, camera='qc', render_profile='realtime', armed=armed)
    result = restart_arguments(args, 'player', 'output')
    assert ('--armed' if armed else '--no-armed') in result


@pytest.fixture
def mesh_fixture(tmp_path):
    # These tests exercise mount transforms. A single triangle exercises the
    # prepared-mesh reader without requiring a separately distributed art pack.
    path = tmp_path / "marker.npz"
    np.savez(path, count=1, **{
        "0_name":"marker","0_body":"right_hand","0_color":[.5,.5,.5],
        "0_roughness":.6,"0_metallic":0.,"0_vertices":[[0.,0,0],[.01,0,0],[0,.01,0]],
        "0_normals":[[0.,0,1]]*3,"0_faces":[[0,1,2]]
    })
    return path


@pytest.mark.parametrize('suffix', ['link', 'Link'])
def test_native_props_follow_wrist_and_keep_handle_at_palm(suffix, mesh_fixture):
    renderer = HandEquipment([f'left_wrist_yaw_{suffix}', f'right_wrist_yaw_{suffix}'], 1, 'cpu', prepared_path=mesh_fixture)
    q = np.column_stack((np.array([[1., 2., 3.], [-2., 4., 1.]]), Rotation.from_euler('xyz', [[.2, .3, .8], [-.5, .9, .2]]).as_quat()))
    original = q.copy()
    mounts = renderer.mounts(q)
    for i, body in enumerate(['shield', 'right_hand']):
        local = SHIELD_HANDLE if i == 0 else np.zeros(3)
        actual = mounts[body][:3, :3] @ local + mounts[body][:3, 3]
        expected = q[i, :3] + Rotation.from_quat(q[i, 3:]).apply([.09, 0., 0.])
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_array_equal(q, original)


def test_equipment_only_change_rebuilds_and_persists(tmp_path, monkeypatch):
    import json

    from omega_playground.ops.runtime import OperationsRuntime

    package = SimpleNamespace(root=tmp_path/'policy', family='sonic', robot={'id': 'unitree_g1'})
    world = SimpleNamespace(package=package)
    teams = SimpleNamespace(players=[], ids=[[0]])
    runtime = OperationsRuntime(world, None, teams, SimpleNamespace(active_scene='campus'), tmp_path)
    runtime.avatar = 'g1'
    runtime.validate_configuration = lambda *args: {'physics_asset': 'prepared'}
    try:
        result = runtime.execute({'kind': 'configure_robot', 'avatar': 'g1', 'armed': True})
        assert result == {'avatar': 'g1', 'armed': True}
        assert json.loads(runtime.reconfigure_request.read_text())['start_idle']
        assert runtime.reconfigure_armed and runtime.paused
    finally:
        runtime.event_file.close()


@pytest.mark.parametrize('avatar', ['a3', 'h2', 'ue4_mannequin', 'ue5_manny'])
def test_soma_prepared_grip_is_local_to_hands(avatar):
    from omega_playground.viewer.armed_skin import load_soma_grip
    root = ROOT/'artifacts/skins/soma23'/avatar
    if not (root/'armed_skin.npz').exists():
        pytest.skip('requires prepared SOMA assets')
    data, original = load_soma_grip(root), np.load(root/'skin.npz')
    np.testing.assert_array_equal(data['bind'], original['bind'])
    names = original['names'].tolist()
    hand_names = ('left_hand', 'right_hand') if avatar in ('a3', 'h2') else ('hand_l', 'hand_r')
    affected = [i for i, n in enumerate(names) if n in hand_names or
                n.startswith(('thumb_', 'index_', 'middle_', 'ring_', 'pinky_'))]
    changed = 0
    for slot in range(int(data['slots'])):
        weights, joints = original[f's{slot}_weights'], original[f's{slot}_joints']
        mask = (weights*np.isin(joints, affected)).sum(axis=1) < 1e-7
        np.testing.assert_allclose(data[f's{slot}_points'][mask], original[f's{slot}_points'][mask], atol=1e-6)
        changed += int(np.sum(np.linalg.norm(data[f's{slot}_points']-original[f's{slot}_points'], axis=1) > .001))
    assert changed > 100
    for mount in data['armed_mounts']:
        np.testing.assert_allclose(mount[:3, :3].T @ mount[:3, :3], np.eye(3), atol=1e-6)
        assert np.linalg.det(mount[:3, :3]) > .999


def test_shield_handle_and_strap_have_anatomical_axes(mesh_fixture):
    renderer = HandEquipment(['left_wrist_yaw_link', 'right_wrist_yaw_link'], 1, 'cpu', prepared_path=mesh_fixture,
                             centers={'l': [.08, -.03, 0], 'r': [.08, .03, 0]})
    q = np.array([[0, 0, 0, 0, 0, 0, 1.], [0, 0, 0, 0, 0, 0, 1.]])
    mount = renderer.mounts(q)['shield']
    # Handle crosses the knuckles (+Z); strap is proximal (-X), not beside the hand.
    np.testing.assert_allclose(mount[:3, :3] @ [1, 0, 0], [0, 0, 1])
    np.testing.assert_allclose(mount[:3, :3] @ [0, 0, 1], [-1, 0, 0])


@pytest.mark.parametrize('flexion', [-1.2, -.6, 0., .6, 1.2])
def test_shield_tracks_forearm_without_slipping_cylindrical_grip(flexion, mesh_fixture):
    names = ['left_elbow_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link']
    renderer = HandEquipment(names, 1, 'cpu', prepared_path=mesh_fixture, centers={'l': [.12, -.03, 0.], 'r': [.12, .03, 0.]})
    q = np.array([[-.24, 0, 0, 0, 0, 0, 1.], [0, 0, 0, 0, 0, 0, 1.], [1, 0, 0, 0, 0, 0, 1.]])
    q[1, 3:] = Rotation.from_euler('z', flexion).as_quat()
    original = q.copy()
    mount = renderer.mounts(q)['shield']
    grip = Rotation.from_quat(q[1, 3:]).apply([.12, -.03, 0.])
    np.testing.assert_allclose(mount[:3, :3] @ SHIELD_HANDLE + mount[:3, 3], grip, atol=1e-12)
    np.testing.assert_allclose(mount[:3, 0], [0, 0, 1.], atol=1e-12)
    forearm = q[0, :3] - grip
    forearm /= np.linalg.norm(forearm)
    assert mount[:3, 2] @ forearm > 0.  # strap still points proximally when wrist clearance requires tilting
    # The entire forearm centerline lies behind the back plate, despite wrist flexion.
    local_elbow = mount[:3, :3].T @ (q[0, :3]-mount[:3, 3])
    local_wrist = mount[:3, :3].T @ (q[1, :3]-mount[:3, 3])
    assert max(local_elbow[1], local_wrist[1]) <= -.055 + 1e-9
    np.testing.assert_array_equal(q, original)
