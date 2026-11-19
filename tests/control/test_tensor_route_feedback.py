import numpy as np
import pytest
import torch

from omega_playground.control.ffpd import FFPD, FFPDConfig
from omega_playground.control.reference import ReferenceTrajectory
from omega_playground.control.route_deviation import LocalRouteProjection
from omega_playground.control.tensor_route_feedback import TensorRouteFeedback


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('count', [1, 3, 9])
def test_entire_feedback_sequence(device, count, monkeypatch):
    reference = ReferenceTrajectory([[0, 0], [20, 0], [20, 20], [0, 20]],
        slot_offsets=[(0., .1*(i%3-1)) for i in range(count)], max_speed=3., max_acceleration=4.)
    config = FFPDConfig(dt=1.)
    gpu = TensorRouteFeedback(reference, list(range(count)), config, device)
    cpu = FFPD(count, config)
    projections = [LocalRouteProjection(reference, i) for i in range(count)]
    for time in np.linspace(-2, reference.duration+3, 120):
        elapsed = time+np.arange(count)*.05
        targets = [reference.sample(t, i) for i,t in enumerate(elapsed)]
        target = np.stack([v[0] for v in targets])
        ahead = np.stack([reference.sample(t+1, i)[0] for i,t in enumerate(elapsed)])
        position = target+np.array([.13, -.17])
        velocity = np.full((count, 2), .05)
        errors = np.stack([p.error(x) for p,x in zip(projections, position)])
        progress = np.array([reference.distance_at(t)-p.reference_distance
                             for p,t in zip(projections, elapsed)])
        expected = cpu.update(target, ahead-target, position, velocity, route_error=errors,
            route_tangent=np.stack([p.tangent for p in projections]), route_progress_error=progress)
        values = [torch.tensor(v, device=device) for v in (position, velocity, elapsed)]
        def forbidden(*args, **kwargs):
            pytest.fail('GPU feedback requested host state')
        with monkeypatch.context() as patch:
            for name in ('cpu', 'numpy', 'item', '__bool__'):
                patch.setattr(torch.Tensor, name, forbidden)
            command, tangent, yaw = gpu.update(*values)
        np.testing.assert_allclose(command.cpu(), expected, atol=1e-10)
        np.testing.assert_allclose(yaw.cpu(), [v[1] for v in targets], atol=1e-10)
        np.testing.assert_allclose(tangent.cpu(), np.stack([p.tangent for p in projections]), atol=1e-10)
