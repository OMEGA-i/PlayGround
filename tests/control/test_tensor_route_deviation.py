from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omega_playground.control.route_deviation import LocalRouteProjection
from omega_playground.control.tensor_route_deviation import TensorRouteProjection


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('count', [1, 3, 9])
def test_local_cursors_match_numpy_at_crossings_and_excursions(device, count):
    t = np.linspace(0, 4*np.pi, 401)
    xy = np.stack((20*np.sin(t), 10*np.sin(2*t)), axis=-1)
    xy[100] = xy[99]  # Degenerate segment must retain the prior tangent.
    reference = SimpleNamespace(xy=xy, yaw=np.arctan2(20*np.cos(2*t), 20*np.cos(t)),
        s=np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))],
        offsets=np.array([[-2*(i//3), i%3-1] for i in range(count)]))
    cpu = [LocalRouteProjection(reference, i) for i in range(count)]
    gpu = TensorRouteProjection(reference, list(range(count)), device)
    for step in range(240):
        positions = np.stack([c.xy[min(400, step+i*2)] for i,c in enumerate(cpu)])
        if 30 <= step < 40:
            positions[0] += 30  # Must not teleport cursor to a remote arc.
        if step == 130:
            selected = list(range(0, count, 2))
            gpu.reset(torch.tensor(selected, device=device))
            for i in selected:
                cpu[i] = LocalRouteProjection(reference, i)
        expected = [c.project(p) for c,p in zip(cpu, positions)]
        error, distance = gpu.project(torch.tensor(positions, device=device))
        np.testing.assert_allclose(error.cpu(), np.stack([r[0] for r in expected]), atol=1e-10)
        np.testing.assert_allclose(distance.cpu(), [r[1] for r in expected], atol=1e-10)
        np.testing.assert_allclose(gpu.progress.cpu(), [c.progress for c in cpu], atol=1e-10)
        np.testing.assert_allclose(gpu.tangent.cpu(), np.stack([c.tangent for c in cpu]), atol=1e-10)
