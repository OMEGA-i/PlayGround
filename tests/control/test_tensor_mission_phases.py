import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omega_playground.control.mission_player import MissionPlayer
from omega_playground.control.tensor_mission_phases import PHASES, TensorMissionPhases


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('goal', [[0., 0.], [2., 2.]])
def test_phases_match_measured_turn_and_stop_sequence(device, goal, monkeypatch):
    spec = {'route': [[0,0], [2,0]], 'legs': [[[0,0],[2,0]], [[2,0],goal]]}
    players = [MissionPlayer(spec, torch.device('cpu')) for _ in range(3)]
    state = TensorMissionPhases(players[0].reference.parts, 3, players[0].config, device)
    positions, velocities, yaws = np.zeros((3,2)), np.zeros((3,2)), np.zeros(3)
    seen = set()
    for now in range(90):
        previous = np.concatenate([p.controller.previous.copy() for p in players])
        ready = np.array([True, now > 5, now > 10])
        tensors = [torch.tensor(v, device=device) for v in (positions, velocities, yaws, previous, ready)]
        def forbidden(*args, **kwargs):
            pytest.fail('Phase transition extracted host state')
        with monkeypatch.context() as patch:
            for name in ('cpu', 'numpy', 'item', '__bool__'):
                patch.setattr(torch.Tensor, name, forbidden)
            state.step(*tensors[:4], float(now), tensors[4])
        for i,p in enumerate(players):
            p.transition_ready = ready[i]
            view = SimpleNamespace(root_pos=torch.tensor([[*positions[i],1.]]),
                root_vel=torch.tensor([[*velocities[i],0.]]),
                root_rot=torch.tensor([[0.,0.,math.sin(yaws[i]/2),math.cos(yaws[i]/2)]], dtype=torch.float64))
            command = p.act(view, None, now)
            seen.add(p.phase)
            assert PHASES[int(state.phase[i].cpu())] == p.phase
            assert int(state.stage[i].cpu()) == p.stage
            np.testing.assert_allclose(state.turn_command[i].cpu(), p.turn_command, atol=1e-10)
            np.testing.assert_allclose(state.leg_started[i].cpu(), p.leg_started, atol=1e-10)
            yaws[i] = math.atan2(command.facing[0,1], command.facing[0,0])
            velocities[i] = command.movement[0,:2].numpy()*max(0., float(command.speed[0]))
            positions[i] += velocities[i]
        if all(p.phase == 'complete' for p in players):
            break
    assert set(PHASES) <= seen
    assert all(p.phase == 'complete' for p in players)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_partial_recovery_preserves_stage_and_completed_players(device):
    p = MissionPlayer({'route': [[0,0],[5,0]], 'legs': [[[0,0],[5,0]], [[5,0],[5,5]]]},
                      torch.device('cpu'))
    state = TensorMissionPhases(p.reference.parts, 3, p.config, device)
    state.stage[:] = 1
    state.phase[:] = torch.tensor([3,3,5], device=device)
    untouched = {k:getattr(state,k)[1].clone() for k in
                 ('stage','phase','anchor','last_act','leg_resume_time','turn_command')}
    ids = torch.tensor([0,2], device=device)
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, device=device)
    state.recover(ids, tensor([[5.,2.],[5.,5.]]), tensor([1.,2.]),
                  tensor([2.5,5.]), tensor([1.4,1.5]), 20.)
    assert state.stage.cpu().tolist() == [1,1,1]
    assert state.phase.cpu().tolist() == [0,3,5]
    for key,value in untouched.items():
        torch.testing.assert_close(getattr(state,key)[1], value, equal_nan=True)
    state.step(tensor([[5.,2.],[5.,3.],[5.,5.]]), tensor([[0.,0.]]*3),
               tensor([1.,1.,2.]), tensor([[0.,0.]]*3), 21.,
               torch.ones(3, dtype=torch.bool, device=device))
    assert float(state.target_yaw[0].cpu()) == 1.4
    assert float(state.leg_resume_time[0].cpu()) == 2.5
    assert int(state.phase[2].cpu()) == 5


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_turn_timeout_and_drift_are_warnings_only(device):
    p = MissionPlayer({'route': [[0,0],[-4,0]]}, torch.device('cpu'))
    state = TensorMissionPhases(p.reference.parts, 1, p.config, device)
    zero = torch.zeros(1,2, dtype=torch.float64, device=device)
    yaw = torch.zeros(1, dtype=torch.float64, device=device)
    ready = torch.ones(1, dtype=torch.bool, device=device)
    for now in range(25):
        state.step(zero, zero, yaw, zero, float(now), ready)
    assert int(state.phase[0].cpu()) == 1
    assert int(state.warning[0].cpu()) == 1
    state.step(zero+1, zero, yaw, zero, 25., ready)
    assert int(state.warning[0].cpu()) == 3
    assert int(state.phase[0].cpu()) == 1


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_retime_keeps_progress_for_different_phases(device):
    spec = {'route': [[0,0],[20,0]], 'legs': [[[0,0],[20,0]], [[20,0],[20,20]]]}
    players = [MissionPlayer(spec, torch.device('cpu')) for _ in range(3)]
    state = TensorMissionPhases(players[0].reference.parts, 3, players[0].config, device)
    state.stage[:] = torch.tensor([0,1,1], device=device)
    state.phase[:] = torch.tensor([3,1,5], device=device)
    state.leg_started[:] = 1.
    state.leg_resume_time[:] = 2.
    for i,p in enumerate(players):
        p.stage = [0,1,1][i]
        p.phase = ['track','turn','complete'][i]
        p.leg_started, p.leg_resume_time = 1., 2.
        p.configure_locomotion('walk', .8, 8.)
    state.retime(players[0].reference.parts, players[0].config, 8.)
    np.testing.assert_allclose(state.leg_started.cpu(), [p.leg_started for p in players], atol=1e-10)
    np.testing.assert_allclose(state.leg_resume_time.cpu(), [p.leg_resume_time for p in players], atol=1e-10)
    assert state.stage.cpu().tolist() == [0,1,1]
    assert state.phase.cpu().tolist() == [3,1,5]
