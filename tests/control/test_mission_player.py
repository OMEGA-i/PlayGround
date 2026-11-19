import math
from types import SimpleNamespace

import numpy as np
import torch

from omega_playground.control.mission_player import MissionPlayer, split_sharp_turns, wrap


def view(position=(0., 0.), yaw=0., velocity=(0., 0.)):
    return SimpleNamespace(root_pos=torch.tensor([[*position, 1.]], dtype=torch.float64),
                           root_vel=torch.tensor([[*velocity, 0.]], dtype=torch.float64),
                           root_rot=torch.tensor([[0., 0., math.sin(yaw/2), math.cos(yaw/2)]], dtype=torch.float64))


def player(route):
    return MissionPlayer({'route': route, 'schedule': {'trigger': 'periodic', 'period_s': .1}},
                         torch.device('cpu'))


def test_sharp_turn_is_split_without_changing_goal_order():
    assert split_sharp_turns([[0, 0], [2, 0], [0, 0]]) == [[[0., 0.], [2., 0.]], [[2., 0.], [0., 0.]]]
    assert len(split_sharp_turns([[0, 0], [2, 0], [0, 1]])) == 2
    assert len(split_sharp_turns([[0, 0], [2, 0], [2, 2]])) == 1


def test_turn_waits_for_measured_heading_and_times_out_without_motion():
    p = player([[0, 0], [-4, 0]])
    headings = []
    for t in np.arange(0., 23., .1):
        cmd = p.act(view(), None, t)
        headings.append(math.atan2(cmd.facing[0, 1], cmd.facing[0, 0]))
        assert float(cmd.speed[0]) <= 0.
        assert p.phase in ('decelerate', 'turn')
    assert np.max(np.abs(headings)) <= math.radians(30) + 1e-6
    assert p.hold['code'] == 'turn_timeout'


def test_measured_turn_slew_then_accelerates_and_reverses_in_place():
    p = player([[0, 0], [2, 0], [0, 0]])
    pos, yaw, velocity = np.zeros(2), 0., np.zeros(2)
    turns = []
    phases = set()
    previous_cmd = np.zeros(2)
    for t in np.arange(0., 80., p.config.dt):
        cmd = p.act(view(pos, yaw, velocity), None, t)
        phases.add(p.phase)
        desired_yaw = math.atan2(cmd.facing[0, 1], cmd.facing[0, 0])
        if p.phase == 'turn':
            assert abs(wrap(desired_yaw-yaw)) <= math.radians(30) + 1e-6
            assert abs(wrap(desired_yaw-yaw)) <= math.radians(30)*p.config.dt + 1e-6
            assert float(cmd.speed[0]) <= 0.
            turns.append(pos.copy())
        yaw = desired_yaw
        velocity = cmd.movement[0, :2].numpy() * max(0., float(cmd.speed[0]))
        assert np.linalg.norm(velocity-previous_cmd) <= p.config.max_acceleration*p.config.dt + 1e-5
        previous_cmd = velocity.copy()
        pos += velocity*p.config.dt
        if p.phase == 'complete':
            break
    assert {'turn', 'decelerate', 'accelerate', 'track', 'settle', 'complete'} <= phases
    assert p.hold is None
    assert np.linalg.norm(pos) < .3
    assert max(p[0] for p in turns) > 1.7
    assert np.max(np.abs(np.array(turns)[:, 1])) < 1e-6


def test_turn_does_not_advance_reference_clock():
    p = player([[0, 0], [-4, 0]])
    for t in np.arange(0., 4., .1):
        p.act(view(), None, t)
    np.testing.assert_allclose(p.targets(100)[0][0], [0, 0])
    assert p.phase == 'turn'


def test_arrival_residual_does_not_prevent_stopping_and_turning():
    p = player([[0, 0], [2, 0], [0, 0]])
    p.phase = 'settle'
    p.phase_started = 0.
    p.anchor = np.array([2., 0.])
    p.turn_command = 0.
    p.initialized = True
    p.controller.previous[:] = [[-.19, 0.]]
    for t in np.arange(.1, 1., .1):
        p.act(view([2.19, 0.]), None, t)
    assert p.phase == 'turn'
    assert p.stage == 1
    assert p.hold is None


def test_run_uses_walk_for_arrival_corrections_without_changing_selected_mode():
    p = player([[0, 0], [40, 0]])
    p.configure_locomotion('run', 1.5, 0.)
    p.phase = 'track'
    p.phase_started = p.leg_started = 0.
    p.initialized = True
    p.anchor = np.zeros(2)
    p.turn_command = 0.
    part = p.reference.parts[0]
    now = part.duration-2.
    target = part.sample(now)[0]
    command = p.act(view(target, velocity=(.9, 0.)), None, now)
    assert int(command.mode[0]) == 2
    assert p.sonic_mode == 'run'
    assert p.phase == 'track'
    # An overshoot outside the arrival tolerance requires a small backward
    # correction, not a backwards Run command against the current facing.
    p.controller.previous[:] = 0.
    command = p.act(view([40.4, 0.]), None, part.duration+.1)
    assert p.phase == 'settle'
    assert int(command.mode[0]) == 2
    assert float(command.movement[0, 0]) < 0
    assert 0 < float(command.speed[0]) <= p.config.max_acceleration*p.config.dt + 1e-6


def test_idle_heading_bias_is_corrected_without_relaxing_arrival_tolerance():
    p = player([[0, 0], [4, 0]])
    yaw = -math.pi/2
    for t in np.arange(0., 30., .1):
        command = p.act(view(yaw=yaw), None, t)
        requested = math.atan2(command.facing[0, 1], command.facing[0, 0])
        if p.phase == 'accelerate':
            assert abs(wrap(yaw)) < math.radians(10)
            break
        # A stable 12-degree learned idle-turn bias would strand an open-loop
        # request outside the unchanged 10-degree arrival criterion.
        yaw += .25*wrap(requested-math.radians(12)-yaw)
    else:
        raise AssertionError('Heading feedback failed to clear a steady idle bias')


def test_centimetre_snap_connector_does_not_demand_southward_initial_turn():
    from omega_playground.control.mission_player import alignment_heading
    p = player([[19.0115242, 4.00985575], [19.0115242, 4.], [30., 4.]])
    part = p.reference.parts[0]
    geometry, timing = part.xy.copy(), part.times.copy()
    assert abs(p.target_yaw) < math.radians(5)
    for t in np.arange(0., 1.1, .1):
        p.act(view((19.0115242, 4.00985575)), None, t)
    assert p.phase in ('accelerate', 'track')
    assert abs(alignment_heading(part, 0.)) < math.radians(5)
    np.testing.assert_array_equal(part.xy, geometry)
    np.testing.assert_array_equal(part.times, timing)
    p.recover_task([0], view((19.0115242, 4.00985575)), 2.)
    assert abs(p.target_yaw) < math.radians(5)


def test_alignment_lookahead_stays_inside_reversal_leg_and_keeps_real_north_turn():
    from omega_playground.control.mission_player import alignment_heading
    p = player([[0., 0.], [2., 0.], [0., 0.]])
    assert abs(alignment_heading(p.reference.parts[0], 0.)) < 1e-6
    assert abs(abs(alignment_heading(p.reference.parts[1], 0.))-math.pi) < 1e-6
    north = player([[0., 0.], [0., 2.], [2., 2.]])
    assert abs(alignment_heading(north.reference.parts[0], 0.)-math.pi/2) < 1e-6


def test_completion_holds_settled_facing_instead_of_rotating_to_route_tangent():
    p = player([[0, 0], [2, 0]])
    p.phase, p.phase_started = 'settle', 0.
    p.anchor = np.array([2., 0.])
    p.turn_command = .2
    p.target_yaw = 0.
    p.initialized = True
    headings = []
    for t in np.arange(.1, 1., .1):
        command = p.act(view([2., 0.], yaw=.18), None, t)
        headings.append(math.atan2(command.facing[0, 1], command.facing[0, 0]))
    assert p.phase == 'complete'
    np.testing.assert_allclose(headings, .2)
