"""CUDA pose ownership and stream ordering at the live skin boundary."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omega_playground.viewer.soma_skin_tensor import TensorSkinUpdater


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_device_input_matches_host_and_owns_snapshot(monkeypatch):
    updater = TensorSkinUpdater.__new__(TensorSkinUpdater)
    updater.painter = SimpleNamespace(num_agents=2, bodies=66)
    updater.stream = torch.cuda.Stream()
    updater.input = torch.empty((2, 66, 7), device='cuda', dtype=torch.float64)
    values = np.arange(132*7, dtype=np.float32).reshape(132, 7)
    updater._copy_input(values)
    updater.stream.synchronize()
    expected = updater.input.clone()
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        source = torch.zeros((132, 14), device='cuda')[:, ::2]
        source.copy_(torch.as_tensor(values, device='cuda'))
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                raise AssertionError('Device poses must not be read back')
            patch.setattr(torch.Tensor, 'cpu', forbidden)
            patch.setattr(torch.Tensor, 'numpy', forbidden)
            updater._copy_input(source)
    updater.stream.synchronize()
    source.zero_()
    torch.cuda.synchronize()
    assert torch.equal(updater.input, expected)
    with pytest.raises(ValueError, match='CUDA device'):
        updater._copy_input(torch.zeros(132, 7))
    with pytest.raises(ValueError, match='float32 or float64'):
        updater._copy_input(torch.zeros(132, 7, device='cuda', dtype=torch.int32))
