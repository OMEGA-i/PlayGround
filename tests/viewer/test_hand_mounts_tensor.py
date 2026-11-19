from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from omega_playground.viewer.hand_equipment import HandMounts
from omega_playground.viewer.hand_mounts_tensor import TensorHandMounts


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('native', [False, True])
def test_mount_batch_matches_clearance_rule_and_degenerate_forearm(device, native):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    rng = np.random.default_rng(51)
    matrices = np.tile(np.eye(4), (32, 3, 1, 1))
    matrices[:, :, :3, :3] = Rotation.random(96, random_state=rng).as_matrix().reshape(32, 3, 3, 3)
    matrices[:, :, :3, 3] = rng.normal(size=(32, 3, 3))
    matrices[0, 0, :3, 3] = matrices[0, 1, :3, 3]  # Zero forearm projection.
    source = HandMounts.__new__(HandMounts)
    source.forearm = 0
    source.hands = {'l': (1, np.eye(3), np.zeros(3)), 'r': (2, np.eye(3), np.zeros(3))}
    source.palette = None if native else SimpleNamespace(bind=np.tile(np.eye(4), (3, 1, 1)))
    expected = []
    for matrix in matrices:
        if native:
            poses = np.concatenate((matrix[:, :3, 3], Rotation.from_matrix(matrix[:, :3, :3]).as_quat()), axis=-1)
        else:
            source.palette.matrices = lambda p, q: matrix
            poses = np.zeros((3, 7))
        expected.append(source.mounts(poses))
    result = TensorHandMounts(source, device).mounts(torch.tensor(matrices, device=device))
    for name, value in result.items():
        np.testing.assert_allclose(value.cpu().numpy(), np.stack([m[name] for m in expected]),
                                   atol=1e-12, rtol=1e-12)
