from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omega_playground.control.route_player import RoutePlayer
from omega_playground.control.tensor_route_player import TensorRoutePlayer


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('lateral', [True, False])
@pytest.mark.parametrize('closed', [True, False])
def test_startup_route_command_and_lifecycle(device, lateral, closed, monkeypatch):
    spec = {'route': [[0,0],[10,0],[10,10],[0,10]], 'reference': {'closed': closed},
            'lateral_control': lateral}
    old, new = RoutePlayer(spec, 1, 'cpu'), TensorRoutePlayer(spec, device)
    for t in range(100):
        xy, _ = old.reference.sample(t-1)
        pos = torch.tensor([[*xy, 1.]], dtype=torch.float64)
        pos[:,1] += .03
        vel = torch.tensor([[.1,.02,0.]], dtype=torch.float64)
        rot = torch.tensor([[0.,0.,0.,1.]], dtype=torch.float64)
        cpu = SimpleNamespace(root_pos=pos, root_vel=vel, root_rot=rot)
        gpu = SimpleNamespace(root_pos=pos.to(device), root_vel=vel.to(device), root_rot=rot.to(device))
        if t == 15:
            for p in (old,new): p.configure_locomotion('walk', .8, t)
        if t == 30:
            old.recover_task([0],cpu,t); new.recover_task([0],gpu,t)
        if t == 75:
            old.recover_idle([0],cpu); new.recover_idle([0],gpu)
        expected = old.act(cpu,None,t)
        def forbidden(*a, **kw): pytest.fail('act read back device state')
        with monkeypatch.context() as patch:
            for name in ('cpu','numpy','item','__bool__'):
                patch.setattr(torch.Tensor,name,forbidden)
            actual = new.act(gpu,None,t)
        for name in ('mode','speed','movement','facing','vertical_speed','height'):
            np.testing.assert_allclose(getattr(actual,name).cpu(),getattr(expected,name),atol=1e-9, err_msg=f"t={t} field={name}")
        # Explicit diagnostic boundary must preserve the previous command.
        new.measure(gpu,t)
