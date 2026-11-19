import numpy as np
import pytest

from omega_playground.control.ffpd import FFPD, FFPDConfig


def feedback(error, velocity, lag=100.):
    controller = FFPD(1, FFPDConfig())
    controller.previous[:] = [.65, 0.]
    commands = []
    for _ in range(20):
        commands.append(controller.update([[lag, 0]], [[.65, 0]], [[0, -error]], [[.65, velocity]],
                                          route_error=[[0, error]], route_tangent=[[1, 0]]))
    return controller, np.array(commands)


def test_centimetre_error_is_corrected_even_under_large_schedule_lag():
    controller, commands = feedback(-.02, 0.)
    assert commands[0, 0, 1] < 0  # immediate intervention, no warning-threshold/deadband gate
    assert commands[-1, 0, 1] == pytest.approx(-.03)
    assert controller.report['lateral_desired_speed'][0] == pytest.approx(-.03)
    assert np.linalg.norm(commands, axis=-1).max() <= 1.+1e-12
    assert np.linalg.norm(np.diff(commands, axis=0), axis=-1).max() <= .1+1e-12
    _, near = feedback(-.02, 0., lag=.1)
    assert near[-1, 0, 1] == pytest.approx(commands[-1, 0, 1])


def test_outward_velocity_is_damped_before_position_deviation_grows():
    controller, commands = feedback(0., .2)
    assert commands[0, 0, 1] < 0
    assert controller.report['lateral_desired_speed'][0] == pytest.approx(-.06)


def test_lateral_limits_and_rotation_invariance():
    rng = np.random.default_rng(7)
    count = 100
    angle = rng.uniform(-np.pi, np.pi, count)
    tangent = np.c_[np.cos(angle), np.sin(angle)]
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    controller = FFPD(count, FFPDConfig())
    previous = controller.previous.copy()
    for _ in range(20):
        cmd = controller.update(100*tangent, tangent, np.zeros((count, 2)), .2*normal,
                                route_error=3*normal, route_tangent=tangent)
        assert np.linalg.norm(cmd, axis=-1).max() <= 1.+1e-12
        assert np.linalg.norm(cmd-previous, axis=-1).max() <= .1+1e-12
        previous = cmd
    np.testing.assert_allclose(np.sum(cmd*normal, axis=-1), .4, atol=1e-12)


def test_invalid_route_feedback_is_rejected():
    controller = FFPD(1, FFPDConfig())
    with pytest.raises(ValueError):
        controller.update([[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]],
                          route_error=[[0, 0]], route_tangent=[[0, 0]])
