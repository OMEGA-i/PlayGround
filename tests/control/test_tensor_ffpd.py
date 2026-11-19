import numpy as np
import pytest
import torch

from omega_playground.control.ffpd import FFPD, FFPDConfig
from omega_playground.control.tensor_ffpd import TensorFFPD


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('count', [1, 3, 9])
@pytest.mark.parametrize('feedback', ['plain', 'lateral', 'progress'])
def test_sequence_matches_numpy_feedback(device, count, feedback):
    rng = np.random.default_rng(731)
    config = FFPDConfig(dt=1.)
    oracle, tensor = FFPD(count, config), TensorFFPD(count, config, device)
    def gpu(value):
        return torch.tensor(value, device=device, dtype=torch.float64)
    for step in range(60):
        states = [rng.normal(size=(count, 3)) for _ in range(4)]
        if step == 30:
            ids = np.arange(0, count, 2)
            oracle.reset(ids, states[-1][ids])
            tensor.reset(torch.tensor(ids, device=device), gpu(states[-1][ids]))
        kwargs = {}
        if feedback != 'plain':
            yaw = rng.uniform(-np.pi, np.pi, count)
            kwargs = dict(route_error=rng.normal(size=(count, 2)),
                          route_tangent=np.stack((np.cos(yaw), np.sin(yaw)), axis=-1))
            if feedback == 'progress':
                kwargs['route_progress_error'] = rng.normal(size=count)
        expected = oracle.update(*states, **kwargs)
        actual = tensor.update(*map(gpu, states), **{k:gpu(v) for k,v in kwargs.items()})
        assert actual.device == tensor.previous.device
        np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-12, rtol=1e-12)
        for key, value in oracle.report.items():
            np.testing.assert_allclose(tensor.report[key].cpu().numpy(), value, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_feedback_does_not_download_or_extract_host_scalars(monkeypatch):
    controller = TensorFFPD(9, FFPDConfig(dt=1.), 'cuda:0')
    values = torch.zeros(9, 2, dtype=torch.float64, device='cuda:0')
    tangent = values.clone()
    tangent[:, 0] = 1.
    def forbidden(*args, **kwargs):
        pytest.fail('Feedback downloaded a tensor or extracted a host scalar')
    with monkeypatch.context() as patch:
        for name in ('cpu', 'numpy', 'item', '__bool__'):
            patch.setattr(torch.Tensor, name, forbidden)
        command = controller.update(values, values, values, values,
            route_error=values, route_tangent=tangent,
            route_progress_error=values[:, 0])
    torch.cuda.synchronize()
    assert torch.equal(command, values)
