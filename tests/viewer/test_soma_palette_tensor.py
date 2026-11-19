import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from omega_playground.viewer.soma_skin import Soma23Palette, SoleSupport
from omega_playground.viewer.soma_palette_tensor import TensorSoma23Palette


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('with_sole', [False, True])
def test_batched_palette_matches_authored_reference(device, with_sole):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    rng = np.random.default_rng(123)
    source = Soma23Palette.__new__(Soma23Palette)
    source.names = ['pelvis', 'unmapped', 'hand']
    source.parents = np.array([-1, 0, 1])
    source.soma_names = ['Hips', 'Hand']
    source.indices = [1, 0]
    source.mapping = {'pelvis': 'Hips', 'hand': 'Hand'}
    source.root = 0
    source.root_offset = np.array([.1, -.2, .03])
    source.bind = np.tile(np.eye(4), (3, 1, 1))
    source.bind[:, :3, :3] = Rotation.random(3, random_state=rng).as_matrix()
    source.bind[:, :3, 3] = rng.normal(size=(3, 3))
    source.inverse = np.linalg.inv(source.bind)
    source.calibration = Rotation.random(3, random_state=rng).as_matrix()
    source.sole = None
    if with_sole:
        s = source.sole = SoleSupport.__new__(SoleSupport)
        s.joints = np.array([[0, 1], [1, 2], [0, 2]])
        weights = np.array([[.3, .7], [.6, .4], [.8, .2]])
        s.weighted_points = np.c_[rng.normal(size=(3, 3)), np.ones(3)][:, None] * weights[..., None]
        s.owners = np.array([0, 1, 0, 1])
        s.corners = rng.normal(size=(4, 3))
    poses = np.concatenate((rng.normal(size=(4, 2, 3)),
                            Rotation.random(8, random_state=rng).as_quat().reshape(4, 2, 4)), -1)
    expected = np.stack([source.matrices(p[:, :3], p[:, 3:]) for p in poses])
    result = TensorSoma23Palette(source, device).matrices(torch.tensor(poses, device=device))
    assert str(result.device) == device
    np.testing.assert_allclose(result.cpu().numpy(), expected, atol=2e-12, rtol=2e-12)
