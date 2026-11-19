"""Authored rigs keep their proportions while physical and visual soles agree."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omega_playground.viewer.soma_skin import Soma23Palette

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=['a3', 'h2', 'ue4_mannequin', 'ue5_manny'])
def skin(request):
    path = ROOT/'artifacts/skins/soma23-native'/request.param/'skin.npz'
    xml = ROOT/'artifacts/policies/soma23-native100k-e20000/robot/soma23.xml'
    if not path.exists() or not xml.exists():
        pytest.skip('SOMA visual assets not installed')
    data = np.load(path)
    return data, Soma23Palette(data, data['soma_names'].tolist(), xml)


def vertices(data, matrices):
    result = []
    for slot in range(int(data['slots'])):
        p = np.c_[data[f's{slot}_points'], np.ones(len(data[f's{slot}_points']))]
        result.append(np.einsum('vkij,vj,vk->vi', matrices[data[f's{slot}_joints']], p,
                                data[f's{slot}_weights'])[:, :3])
    return np.concatenate(result)


def test_actual_rest_mesh_sole_not_ankle_matches_physics(skin):
    data, rig = skin
    positions = data['soma_rest'].copy()
    rotations = np.tile([0., 0., 0., 1.], (len(positions), 1))
    before = positions.copy()
    matrices = rig.matrices(positions, rotations)
    mesh = vertices(data, matrices)
    assert mesh[:, 2].min() == pytest.approx(rig.sole.physical_height(positions, rotations), abs=2e-6)
    np.testing.assert_array_equal(positions, before)
    # Full mesh gets one translation: no new limb scaling or changed articulation.
    authored = rig.targets(positions, rotations) @ rig.inverse
    np.testing.assert_allclose(matrices[:, :3, :3], authored[:, :3, :3], atol=1e-12)
    np.testing.assert_allclose(np.diff(matrices[:, :3, 3], axis=0),
                               np.diff(authored[:, :3, 3], axis=0), atol=1e-12)


def test_world_heading_translation_and_jump_are_preserved(skin):
    data, rig = skin
    positions = data['soma_rest'].copy()
    quat = np.tile([0., 0., 0., 1.], (len(positions), 1))
    original = rig.matrices(positions, quat)
    r = Rotation.from_euler('z', 1.4)
    shift = np.array([10., -7., .8])
    moved = rig.matrices(positions @ r.as_matrix().T + shift, (r * Rotation.from_quat(quat)).as_quat())
    np.testing.assert_allclose(moved[:, :3, 3], original[:, :3, 3] @ r.as_matrix().T + shift, atol=1e-7)
    np.testing.assert_allclose(moved[:, :3, :3], r.as_matrix() @ original[:, :3, :3], atol=1e-7)


def test_recorded_motion_keeps_support_and_bone_lengths(skin):
    _, rig = skin
    path = ROOT/'artifacts/qc/soma_mission_filtered/poses.npz'
    if not path.exists():
        pytest.skip('Recorded SOMA walk not installed')
    poses = np.load(path)['body_q'][::90, :23]
    for pose in poses:
        matrices = rig.matrices(pose[:, :3], pose[:, 3:])
        assert rig.sole.visual_height(matrices) == pytest.approx(
            rig.sole.physical_height(pose[:, :3], pose[:, 3:]), abs=1e-7)
        targets = matrices @ rig.bind
        for child, parent in enumerate(rig.parents):
            if parent >= 0:
                assert np.linalg.norm(targets[child, :3, 3] - targets[parent, :3, 3]) == pytest.approx(
                    np.linalg.norm(rig.bind[child, :3, 3] - rig.bind[parent, :3, 3]), abs=1e-7)


def test_elbow_calibration_uses_bend_without_axial_roll(skin):
    _, rig = skin
    reverse = {body: bone for bone, body in rig.mapping.items()}
    for side in ('Left', 'Right'):
        arm, forearm, hand = [rig.names.index(reverse[side+p]) for p in ('Arm', 'ForeArm', 'Hand')]
        upper_delta = rig.calibration[arm] @ np.linalg.inv(rig.bind[arm, :3, :3])
        lower_delta = rig.calibration[forearm] @ np.linalg.inv(rig.bind[forearm, :3, :3])
        before = upper_delta @ (rig.bind[hand, :3, 3]-rig.bind[forearm, :3, 3])
        after = rig.rest[side+'Hand']-rig.rest[side+'ForeArm']
        angle = np.arccos(np.clip(before @ after / np.linalg.norm(before) / np.linalg.norm(after), -1., 1.))
        assert Rotation.from_matrix(lower_delta @ upper_delta.T).magnitude() == pytest.approx(angle, abs=1e-7)


def test_all_avatars_have_same_anatomical_palm_frame(skin):
    _, rig = skin
    reverse = {body: bone for bone, body in rig.mapping.items()}
    for side, sign in [('Left', 1.), ('Right', -1.)]:
        name = reverse[side+'Hand']
        i = rig.names.index(name)
        frame = rig.calibration[i] @ np.linalg.inv(rig.bind[i, :3, :3]) @ rig.palm_frames[name]
        np.testing.assert_allclose(frame[:, 0], [sign, 0., 0.], atol=1e-7)
        np.testing.assert_allclose(frame[:, 1], [0., -1., 0.], atol=1e-7)
        assert np.linalg.det(frame) == pytest.approx(1., abs=1e-7)
