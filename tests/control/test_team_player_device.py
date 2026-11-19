from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omega_playground.control.teams import TaskControllerBatch
from omega_playground.policy.base import RobotBatchView, JoystickBatch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU Player dispatch')
def test_gpu_player_batch_never_builds_host_view(monkeypatch):
    device=torch.device('cuda:0')
    tensor=lambda *shape: torch.zeros(*shape,device=device)
    view=RobotBatchView(tensor(1,1,3),tensor(1,1,4),tensor(1,1,3),tensor(1,1,3),
                       tensor(1,2),tensor(1,2),tensor(1))
    def act(selected, task, now):
        assert selected.root_pos.device==device
        assert selected.body_pos.data_ptr()==view.body_pos.data_ptr()
        return JoystickBatch(torch.zeros(1,dtype=torch.long,device=device),tensor(1),tensor(1,3),
                             tensor(1,3),tensor(1),tensor(1))
    batch=TaskControllerBatch.__new__(TaskControllerBatch)
    batch.players=[SimpleNamespace(device=device,act=act)]
    batch.ids=[np.array([0])]
    batch.device=device
    monkeypatch.setattr('omega_playground.control.teams.host_view',
                        lambda v: pytest.fail('GPU Player was routed through CPU'))
    assert batch.act(view,None,0.).movement.device==device


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU startup route')
def test_startup_route_selects_gpu_without_host_dispatch(monkeypatch):
    monkeypatch.delenv('OMEGA_GPU_PLAYER', raising=False)
    batch = TaskControllerBatch([{'route': [[0,0],[10,0]],
        'reference': {'slot_offsets': [[0,0]]}}], 'cuda:0', None, 'sonic')
    assert type(batch.players[0]).__name__ == 'TensorRoutePlayer'
    assert str(batch.players[0].device) == 'cuda:0'
    tensor = lambda *shape: torch.zeros(*shape, device='cuda:0')
    view = RobotBatchView(tensor(1,1,3),tensor(1,1,4),tensor(1,1,3),tensor(1,1,3),
                         tensor(1,2),tensor(1,2),tensor(1))
    monkeypatch.setattr('omega_playground.control.teams.host_view',
                        lambda v: pytest.fail('Startup GPU route requested CPU view'))
    assert batch.act(view,None,0.).movement.is_cuda


@pytest.mark.parametrize('override,expected', [(None, 'cuda:0'), ('0', 'cpu'), ('1', 'cuda:0')])
def test_service_startup_device_selection_without_cuda(monkeypatch, override, expected):
    if override is None:
        monkeypatch.delenv('OMEGA_GPU_PLAYER', raising=False)
    else:
        monkeypatch.setenv('OMEGA_GPU_PLAYER', override)
    seen = []
    def factory(family, spec, count, device):
        seen.append(str(device))
        return SimpleNamespace(schedule=1)
    monkeypatch.setattr('omega_playground.control.teams.route_player', factory)
    TaskControllerBatch([{'reference': {'slot_offsets': [[0, 0]]}}], 'cuda:0', None)
    assert seen == [expected]
