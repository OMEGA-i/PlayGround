from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip('warp')
pytest.importorskip('mujoco_warp')
from omega_playground.viewer.mjwarp import MjWarpPoseRenderer


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_camera_update_preserves_other_cameras_without_readback(device, monkeypatch):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA required')
    positions = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    rotations = np.arange(54, dtype=np.float32).reshape(2, 3, 3, 3)
    renderer = MjWarpPoseRenderer.__new__(MjWarpPoseRenderer)
    renderer.device = device
    renderer.data = SimpleNamespace(cam_xpos=wp.array(positions, dtype=wp.vec3, device=device),
                                    cam_xmat=wp.array(rotations, dtype=wp.mat33, device=device))
    eye, target = np.array([2., 3., 4.], np.float32), np.array([1., 0., 1.], np.float32)
    z = eye - target
    x = np.cross([0., 0., 1.], z)
    z /= np.linalg.norm(z)
    x /= np.linalg.norm(x)
    positions[0, 0] = eye
    rotations[0, 0] = np.stack((x, np.cross(z, x), z), axis=1)
    with monkeypatch.context() as patch:
        patch.setattr(wp.array, 'numpy', lambda self: pytest.fail('Camera update read back GPU data'))
        renderer.look_at(eye, target)
    np.testing.assert_array_equal(renderer.data.cam_xpos.numpy(), positions)
    np.testing.assert_array_equal(renderer.data.cam_xmat.numpy(), rotations)
