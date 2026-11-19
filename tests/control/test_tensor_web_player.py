import math
from types import SimpleNamespace

import numpy as np
import torch

from omega_playground.control.mission_player import MissionPlayer
from omega_playground.control.tensor_web_player import TensorWebPlayer
from omega_playground.control.player_factory import mission_player


def test_web_diagnostics_recovery_and_new_task_history(monkeypatch):
    monkeypatch.setenv('OMEGA_GPU_PLAYER','1')
    spec={'route': [[0,0],[8,0]], 'legs': [[[0,0],[8,0]],[[8,0],[8,8]]]}
    cpu=MissionPlayer(spec,torch.device('cpu'))
    player=mission_player('sonic',spec,torch.device('cpu'))
    assert isinstance(player,TensorWebPlayer)
    # Web task replacement preserves acceleration history.
    for p in (cpu,player):
        p.controller.previous[:]=[.3,0.]
        p.initialized=True
        p.safety_holds=p.tracking_divergence_hold=False
    player.status()  # Publish may precede the first scheduled command.
    np.testing.assert_array_equal(player.controller.previous, [[.3,0.]])
    position,velocity,yaw=np.zeros(2),np.zeros(2),0.
    for now in range(100):
        view=SimpleNamespace(root_pos=torch.tensor([[*position,1.]]),
            root_vel=torch.tensor([[*velocity,0.]]),
            root_rot=torch.tensor([[0.,0.,math.sin(yaw/2),math.cos(yaw/2)]],dtype=torch.float64))
        if now==6:
            for p in (cpu,player):p.recover_task([0],view,now)
        if now==12:
            for p in (cpu,player):p.configure_locomotion('walk',.8,now)
        expected,actual=cpu.act(view,None,now),player.act(view,None,now)
        for field in ('mode','speed','movement','facing'):
            np.testing.assert_allclose(getattr(actual,field),getattr(expected,field),atol=1e-8)
        for p in (cpu,player):
            p.check(view,now)
            p.status()
        assert player.status()==cpu.status()
        yaw=math.atan2(expected.facing[0,1],expected.facing[0,0])
        velocity=expected.movement[0,:2].numpy()*max(0.,float(expected.speed[0]))
        position+=velocity
        if cpu.phase=='complete':break
    assert cpu.phase=='complete'
    player.recover_idle([0],view)
    idle=player.act(view,None,now+1)
    assert int(idle.mode[0])==0 and float(idle.speed[0])==-1
    assert player.status()['phase']=='idle'


def test_aligned_start_requires_stationary_heading_and_preserves_default():
    import pytest
    view = SimpleNamespace(root_pos=torch.tensor([[0.,0.,1.]]),
        root_vel=torch.zeros(1,3), root_rot=torch.tensor([[0.,0.,0.,1.]]))
    player = TensorWebPlayer({'route': [[0,0],[8,0]]}, 'cpu')
    player.reset([0],view)
    assert player.status()['phase'] == 'decelerate'
    view.root_vel[0,0] = .2
    with pytest.raises(ValueError, match='stationary'):
        player.start_aligned(view)
    view.root_vel.zero_()
    view.root_rot[:] = torch.tensor([[0.,0.,1.,0.]])
    with pytest.raises(ValueError, match='heading'):
        player.start_aligned(view)
    view.root_rot[:] = torch.tensor([[0.,0.,0.,1.]])
    player.start_aligned(view)
    assert player.status()['phase'] == 'accelerate'
    assert float(player.act(view,None,1.).speed[0]) > 0
