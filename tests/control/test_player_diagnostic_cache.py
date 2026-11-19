from types import SimpleNamespace

import pytest
import torch

from omega_playground.control.tensor_route_player import TensorRoutePlayer
from omega_playground.control.tensor_web_player import TensorWebPlayer


@pytest.mark.parametrize('player_type', [TensorRoutePlayer, TensorWebPlayer])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_diagnostic_reads_only_after_mutation(player_type, device, monkeypatch):
    player = player_type({'route': [[0,0],[10,0]]}, device)
    view = SimpleNamespace(root_pos=torch.tensor([[0.,0.,1.]],device=device),
                           root_vel=torch.zeros(1,3,device=device),
                           root_rot=torch.tensor([[0.,0.,0.,1.]],device=device))
    reads=[]
    original=torch.Tensor.cpu
    def counted(tensor,*args,**kwargs):
        reads.append(tensor.numel())
        return original(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'cpu',counted)
    player.act(view,None,0.)
    player._sync()
    assert reads
    reads.clear()
    for _ in range(50):player._sync()
    assert not reads
    events=[lambda:player.act(view,None,1.),
            lambda:player.reset(torch.tensor([0],device=device),view),
            lambda:player.recover_task([0],view,2.),
            lambda:player.configure_locomotion('walk',.8,2.),
            lambda:player.recover_idle([0],view)]
    for event in events:
        event()
        player._sync()
        assert reads, 'A mutation must invalidate the diagnostic snapshot'
        reads.clear()
        player._sync()
        assert not reads, 'Unchanged diagnostic snapshot must not synchronize again'
