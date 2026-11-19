from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from omega_playground.control.mission_player import MissionPlayer
from omega_playground.control.route_player import RoutePlayer
from omega_playground.ops.runtime import OperationsRuntime
from omega_playground.policy.base import RobotBatchView


def view(xy, yaw=0):
    n = len(xy)
    pos = torch.tensor([[*p, 1.] for p in xy])[:, None]
    rot = torch.tensor([0., 0., np.sin(yaw/2), np.cos(yaw/2)]).repeat(n, 1, 1)
    return RobotBatchView(pos, rot, torch.zeros_like(pos), torch.zeros_like(pos),
                          torch.zeros(n, 1), torch.zeros(n, 1), torch.zeros(n))


def spec():
    return {'route': [[0, 0], [100, 0]], 'schedule': {'trigger': 'periodic', 'period_s': .1}}


def test_speed_lag_is_not_route_deviation_and_lateral_error_is():
    player = RoutePlayer(spec(), 1, 'cpu')
    player.safety_holds = False
    for t in (0., 90., 100.):
        report = player.check(view([[2., .4]]), t)
    assert np.linalg.norm(report['position_error']) > 50
    np.testing.assert_allclose(report['route_error'], [[0., -.4]], atol=1e-7)
    assert player.hold is None and not player.warnings
    player.check(view([[2., 3.]]), 101.)
    player.check(view([[2., 3.]]), 103.)
    assert player.warnings[0]['error_m'] == [3.]


def test_local_window_does_not_snap_to_later_parallel_segment():
    player = RoutePlayer({**spec(), 'route': [[0, 0], [30, 0], [30, 6], [0, 6]]}, 1, 'cpu')
    # The later return road is only 0.1 m away, but not the active local route.
    error = player.measure(view([[1., 5.9]]), 90.)['route_error']
    np.testing.assert_allclose(error, [[0., -5.9]], atol=1e-6)


def test_mission_reversal_uses_active_leg():
    player = MissionPlayer({**spec(), 'route': [[0, 0], [10, 0], [0, 0]]}, 'cpu')
    player.stage, player.phase = 1, 'track'
    np.testing.assert_allclose(player.measure(view([[8., .5]]), 100.)['route_error'], [[0., -.5]])


@pytest.mark.parametrize('code', ['robot_contact', 'hard_separation', 'fallen_in_formation',
                                  'contact_capacity', 'turn_timeout', 'stop_timeout', 'turn_drift', 'off_navigable'])
def test_operational_safety_only_warns(tmp_path, code):
    runtime = OperationsRuntime(SimpleNamespace(sim_time=1, physics_steps=200), None,
                                SimpleNamespace(players=[], ids=[]), SimpleNamespace(event=Mock()), tmp_path)
    try:
        runtime.paused = False
        runtime.report_safety({'code': code, 'agents': [0]})
        runtime.report_safety({'code': code, 'agents': [0]})
        assert not runtime.paused and runtime.hold is None
        assert runtime.server.event.call_count == 1
    finally:
        runtime.close()


def test_nonfinite_is_a_computation_failure(tmp_path):
    runtime = OperationsRuntime(SimpleNamespace(sim_time=1, physics_steps=200), None,
                                SimpleNamespace(players=[], ids=[]), SimpleNamespace(event=Mock()), tmp_path)
    try:
        runtime.report_safety({'code': 'nonfinite'})
        assert runtime.paused and runtime.hold['code'] == 'nonfinite'
    finally:
        runtime.close()



def test_mission_recovery_retains_later_leg_and_rebases_stale_clock():
    player = MissionPlayer({**spec(), 'legs': [[[0, 0], [10, 0]], [[10, 0], [0, 0]],
                                            [[0, 0], [0, 10]]]}, 'cpu')
    state = view([[7., 0.]], yaw=np.pi)
    player.stage, player.phase = 1, 'track'
    player.recover_task([0], state, 100.)
    assert player.stage == 1 and len(player.reference.parts) == 3
    assert player.status()['target'] == [0., 0.]
    assert player.phase == 'decelerate'
    np.testing.assert_allclose(player.targets(100.)[0][0], [7., 0.])
    for t in (100., 100.4, 100.8):
        player.act(state, None, t)
    assert player.phase == 'accelerate'
    np.testing.assert_allclose(player.targets(100.8)[0][0], [7., 0.], atol=1e-6)
    assert player.targets(101.8)[0][0][0] < 7.
    assert player.stage == 1


def test_completed_mission_is_not_restarted_by_recovery():
    player = MissionPlayer(spec(), 'cpu')
    player.phase = 'complete'
    state = view([[100., 0.]])
    player.recover_task([0], state, 200.)
    assert player.phase == 'complete'
    assert player.act(state, None, 201.).mode.tolist() == [0]


@pytest.mark.parametrize('phase,position,code', [('turn', [0., 0.], 'turn_timeout'),
                                                ('turn', [1., 0.], 'turn_drift'),
                                                ('settle', [5., 0.], 'stop_timeout')])
def test_mission_controller_warnings_never_latch_hold(tmp_path, phase, position, code):
    player = MissionPlayer(spec(), 'cpu')
    runtime = OperationsRuntime(SimpleNamespace(sim_time=25., physics_steps=5000), None,
                                SimpleNamespace(players=[player], ids=[np.array([0])]),
                                SimpleNamespace(event=Mock()), tmp_path)
    try:
        state = view([position], yaw=1.)
        player.phase, player.phase_started, player.turn_command = phase, 0., 1.
        player.act(state, None, 25.)
        for t in (25., 25.02):
            player.check(state, t)
            runtime.publish_warnings()
            assert player.hold is None
            assert any(w['code'] == code for w in player.warnings)
        assert runtime.server.event.call_count == 1
    finally:
        runtime.close()


def test_route_recovery_rebases_even_when_old_reference_clock_has_expired():
    player = RoutePlayer(spec(), 1, 'cpu')
    state = view([[7., 0.]])
    player.recover_task([0], state, 1000.)
    np.testing.assert_allclose(player.targets(1000.)[0][0], [7., 0.])
    assert player.act(state, None, 1000.1).mode.tolist() == [2]


@pytest.mark.parametrize('mode,expected', [('in_place', [7., 2.]), ('task_start', [3., 2.])])
def test_recovery_mode_uses_saved_task_start_not_spawn(tmp_path, mode, expected):
    player = MissionPlayer({**spec(), 'legs': [[[3, 2], [10, 2]], [[10, 2], [20, 2]]]}, 'cpu')
    player.stage, player.phase = 1, 'track'
    state = view([[7., 2.]], yaw=.7)
    def reset(ids, xy, yaw):
        state.body_pos[ids, 0, :2] = torch.as_tensor(xy, dtype=state.body_pos.dtype)
        return {}
    world = SimpleNamespace(sim_time=12., physics_steps=2400, torch_device='cpu',
                            robot_view=lambda: state, reset_agents=Mock(side_effect=reset))
    runner = SimpleNamespace(wbc=Mock(), joystick=Mock(), request_reset=Mock(), close=Mock())
    runtime = OperationsRuntime(world, runner, SimpleNamespace(players=[player], ids=[np.array([0])]),
                                SimpleNamespace(event=Mock()), tmp_path)
    runtime.task_origins[0] = (np.array([[3., 2.]]), np.array([.4]))
    try:
        audit = runtime.execute({'kind': 'reset', 'recovery_mode': mode})
        np.testing.assert_allclose(world.reset_agents.call_args.args[1], [expected])
        assert player.stage == (0 if mode == 'task_start' else 1)
        assert audit['recovery_mode'] == mode
        assert runtime.paused
        with pytest.raises(ValueError, match='recovery mode'):
            runtime.execute({'kind': 'reset', 'recovery_mode': 'invalid'})
        assert world.reset_agents.call_count == 1
    finally:
        runtime.close()
