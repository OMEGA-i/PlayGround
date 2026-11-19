import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omega_playground.control.mission_player import MissionPlayer
from omega_playground.control.tensor_mission_player import TensorMissionPlayer
from omega_playground.control.tensor_mission_phases import PHASES


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('goal', [[0.,0.], [2.,2.]])
@pytest.mark.parametrize('mode', ['walk', 'run'])
def test_full_command_sequence_matches_cpu(device, goal, mode, monkeypatch):
    spec = {'route': [[0,0],[2,0]], 'legs': [[[0,0],[2,0]],[[2,0],goal]], 'sonic_mode':mode}
    players = [MissionPlayer(spec, torch.device('cpu')) for _ in range(3)]
    gpu = TensorMissionPlayer(spec, device, count=3)
    positions, velocities, yaw = np.zeros((3,2)), np.zeros((3,2)), np.zeros(3)
    for now in range(90):
        rotations = np.stack((np.zeros(3),np.zeros(3),np.sin(yaw/2),np.cos(yaw/2)),axis=-1)
        cpu_view = SimpleNamespace(root_pos=torch.tensor(np.c_[positions,np.ones(3)]),
            root_vel=torch.tensor(np.c_[velocities,np.zeros(3)]), root_rot=torch.tensor(rotations))
        gpu_view = SimpleNamespace(**{k:v.to(device) for k,v in vars(cpu_view).items()})
        ready = np.array([True, now>5, now>10])
        gpu.transition_ready = torch.tensor(ready, device=device)
        def forbidden(*args, **kwargs):
            pytest.fail('Player requested CPU state during command calculation')
        with monkeypatch.context() as patch:
            for name in ('cpu','numpy','item','__bool__'):
                patch.setattr(torch.Tensor,name,forbidden)
            actual = gpu.act(gpu_view,None,float(now))
        for i,p in enumerate(players):
            p.transition_ready=ready[i]
            expected=p.act(SimpleNamespace(**{k:v[i:i+1] for k,v in vars(cpu_view).items()}),None,now)
            for field in ('mode','speed','movement','facing','vertical_speed','height'):
                np.testing.assert_allclose(getattr(actual,field)[i:i+1].cpu(),getattr(expected,field),atol=1e-9)
            assert PHASES[int(gpu.state.phase[i].cpu())]==p.phase
            yaw[i]=math.atan2(expected.facing[0,1],expected.facing[0,0])
            velocities[i]=expected.movement[0,:2].numpy()*max(0.,float(expected.speed[0]))
            positions[i]+=velocities[i]
        if all(p.phase=='complete' for p in players):break
    assert all(p.phase=='complete' for p in players)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_recover_and_retime_command_sequence(device):
    spec={'route':[[0,0],[8,0]], 'legs': [[[0,0],[8,0]], [[8,0],[8,8]]]}
    cpu=MissionPlayer(spec,torch.device('cpu'))
    gpu=TensorMissionPlayer(spec,device)
    position,velocity,yaw=np.zeros(2),np.zeros(2),0.
    for now in range(100):
        view=SimpleNamespace(root_pos=torch.tensor([[*position,1.]]),
            root_vel=torch.tensor([[*velocity,0.]]),
            root_rot=torch.tensor([[0.,0.,math.sin(yaw/2),math.cos(yaw/2)]],dtype=torch.float64))
        device_view=SimpleNamespace(**{k:v.to(device) for k,v in vars(view).items()})
        if now==6:
            cpu.recover_task([0],view,now)
            gpu.recover_task([0],device_view,now)
        if now==12:
            cpu.configure_locomotion('walk',.8,now)
            gpu.configure_locomotion('walk',.8,now)
        expected,actual=cpu.act(view,None,now),gpu.act(device_view,None,now)
        for field in ('mode','speed','movement','facing'):
            np.testing.assert_allclose(getattr(actual,field).cpu(),getattr(expected,field),atol=1e-8)
        assert PHASES[int(gpu.state.phase[0].cpu())]==cpu.phase
        yaw=math.atan2(expected.facing[0,1],expected.facing[0,0])
        velocity=expected.movement[0,:2].numpy()*max(0.,float(expected.speed[0]))
        position+=velocity
        if cpu.phase=='complete':break
    assert cpu.phase=='complete'


@pytest.mark.parametrize('player_type', [MissionPlayer, TensorMissionPlayer])
def test_new_opposite_trajectory_never_advances_before_measured_alignment(player_type):
    p = player_type({'route': [[0, 0], [-4, 0]]}, torch.device('cpu'))
    def view(yaw):
        return SimpleNamespace(root_pos=torch.tensor([[0., 0., 1.]], dtype=torch.float64),
            root_vel=torch.zeros(1, 3, dtype=torch.float64),
            root_rot=torch.tensor([[0., 0., math.sin(yaw/2), math.cos(yaw/2)]], dtype=torch.float64))
    # Even a turn timeout must not bypass the measured-heading gate.
    for now in range(30):
        command = p.act(view(0.), None, float(now))
        assert command.speed[0] <= 0
        torch.testing.assert_close(command.movement, torch.zeros_like(command.movement))
    commands = [p.act(view(math.pi), None, float(now)) for now in range(30, 34)]
    assert any(command.speed[0] > 0 for command in commands)
