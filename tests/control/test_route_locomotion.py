import numpy as np
import pytest
import torch

from omega_playground.control.locomotion import MODES
from omega_playground.control.mission_player import MissionPlayer
from omega_playground.control.route_player import RoutePlayer


def player(mission=False):
    spec = {'route': [[0,0],[30,0],[30,20]], 'schedule': {'trigger':'periodic','period_s':.1}}
    return MissionPlayer(spec, torch.device('cpu')) if mission else RoutePlayer(spec, 1, torch.device('cpu'))


@pytest.mark.parametrize('mission', [False, True])
def test_speed_change_keeps_geometry_progress_and_limiter(mission):
    p = player(mission)
    if mission:
        p.phase, p.leg_started = 'track', 1.
    now = 12.
    before = p.targets(now)
    geometry = p.reference.xy.copy()
    p.controller.previous[:] = [[.6,.1]]
    p.configure_locomotion('run', 1.5, now)
    np.testing.assert_allclose(p.targets(now)[0][0], before[0][0], atol=1e-8)
    np.testing.assert_array_equal(p.reference.xy, geometry)
    np.testing.assert_array_equal(p.controller.previous, [[.6,.1]])
    assert p.sonic_mode == 'run'
    assert p.reference.nominal_speed == 1.5
    assert p.config.max_speed > 1.5
    p.configure_locomotion('walk', .4, now)
    np.testing.assert_allclose(p.targets(now)[0][0], before[0][0], atol=1e-8)


def test_only_walk_run_and_invalid_change_is_atomic():
    p = player(True)
    original = p.reference
    assert set(MODES) == {'walk', 'run'}
    for mode,speed in [('idle',1), ('walk',0), ('run',float('nan')), ('run',4)]:
        with pytest.raises(ValueError):
            p.configure_locomotion(mode, speed, 0)
        assert p.reference is original and p.sonic_mode == 'walk'


def test_waiting_and_completed_task_do_not_restart():
    p = player()
    p.configure_locomotion('run',1.5,0)
    assert p.start_times[0] == 1.
    now = 1000.
    end = p.targets(now)
    p.configure_locomotion('walk',.4,now)
    np.testing.assert_allclose(p.targets(now)[0][0],end[0][0])
    m = player(True)
    m.phase='complete'
    m.configure_locomotion('run',1.5,now)
    assert m.phase == 'complete'


def test_retimed_route_preserves_curvature_speed_limits_and_inverse_clock():
    p = player()
    old = p.reference
    new = old.retimed(1.5,1.85,1.)
    assert np.max(new.speed) > 1.
    assert np.max(new.curvature*new.speed**2) <= .500001
    for t in np.linspace(0,new.duration,101):
        assert abs(new.time_at_distance(new.distance_at(t))-t) < 1e-5
    assert old.nominal_speed == .65


def test_runtime_changes_settings_without_replacing_task_or_resetting(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from omega_playground.ops.runtime import OperationsRuntime

    p = player(True)
    p.phase, p.leg_started = 'track', 1.
    world = SimpleNamespace(package=SimpleNamespace(family='sonic'), sim_time=12., physics_steps=2400, torch_device=torch.device('cpu'),
                            robot_view=lambda: SimpleNamespace(root_pos=torch.tensor([[4.,0.,1.]]), root_rot=torch.tensor([[0.,0.,0.,1.]])),
                            reset_agents=Mock())
    nav = SimpleNamespace(plan_goal=Mock(return_value={'route': [[4,0],[10,0]], 'clearance_m': .6}))
    teams = SimpleNamespace(players=[p], ids=[np.array([0])])
    runtime = OperationsRuntime(world, None, teams, SimpleNamespace(event=Mock(),navigation=nav), tmp_path)
    monkeypatch.setattr('omega_playground.ops.runtime.validate_reference', lambda *args: None)
    try:
        runtime.execute({'kind': 'locomotion', 'mode': 'run', 'speed': 1.5})
        assert teams.players[0] is p and p.phase == 'track'
        assert p.locomotion() == {'mode': 'run', 'speed': 1.5}
        with pytest.raises(ValueError):
            runtime.execute({'kind': 'manual', 'mode': 'run', 'speed': 1.5, 'heading': 90})
        assert teams.players[0].locomotion() == {'mode':'run','speed':1.5}
        world.reset_agents.assert_not_called()
    finally:
        runtime.close()
