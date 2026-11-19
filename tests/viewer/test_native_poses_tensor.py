import numpy as np
import pytest
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from omega_playground.viewer.mjwarp_armed import ArmedPoseAdapter
from omega_playground.viewer.native_poses_tensor import NativeDevicePoses


@pytest.mark.parametrize('armed', [False, True])
def test_native_device_poses_match_cpu_mounts(armed):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    wp.init()
    rng = np.random.default_rng(34)
    meta = {'num_agents': 3, 'policy_body_names': ['left_elbow_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link']}
    if armed:
        meta['armed'] = {'centers': {'l': [.09, 0., 0.], 'r': [.08, .01, 0.]}}
    adapter = NativeDevicePoses(meta, 'cuda:0')
    for _ in range(3):
        poses = np.concatenate((rng.normal(size=(9, 3)), Rotation.random(9, random_state=rng).as_quat()), axis=1).astype(np.float32)
        expected = ArmedPoseAdapter(meta).extend(poses) if armed else poses
        actual = adapter.update(torch.tensor(poses, device='cuda')).numpy()
        np.testing.assert_array_equal(actual[:9], poses)
        np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(Rotation.from_quat(actual[:, 3:]).as_matrix(),
                                   Rotation.from_quat(expected[:, 3:]).as_matrix(), atol=2e-6, rtol=2e-6)
