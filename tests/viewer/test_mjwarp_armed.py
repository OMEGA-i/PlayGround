import numpy as np

from omega_playground.viewer.mjwarp_armed import ArmedPoseAdapter


def test_equipment_follows_each_agents_authoritative_pose_without_mutation():
    names = ['left_wrist_yaw_link', 'right_wrist_yaw_link', 'left_elbow_link']
    meta = {'policy_body_names': names, 'num_agents': 2,
            'armed': {'centers': {'l': [.12, -.03, 0.], 'r': [.12, .03, 0.]}}}
    adapter = ArmedPoseAdapter(meta)
    pose = np.zeros((6, 7), np.float32)
    pose[:, 6] = 1.
    pose[:, 2] = 1.
    pose[2, 0] = pose[5, 0] = -.2
    pose[3:, 1] = 3.
    before = pose.copy()
    result = adapter.extend(pose)
    np.testing.assert_array_equal(pose, before)
    np.testing.assert_array_equal(result[:6], pose)
    assert result.shape == (10, 7)
    np.testing.assert_allclose(result[8:, :3]-result[6:8, :3], [[0., 3., 0.]]*2, atol=1e-6)
    shifted = pose.copy()
    shifted[:3, :3] += [2., 1., .5]
    moved = adapter.extend(shifted)
    np.testing.assert_allclose(moved[6:8, :3]-result[6:8, :3], [[2., 1., .5]]*2, atol=1e-6)
    np.testing.assert_array_equal(moved[8:], result[8:])
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:], axis=-1), 1., atol=1e-6)
