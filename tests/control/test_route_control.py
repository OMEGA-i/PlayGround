from pathlib import Path

import numpy as np
import pytest

from omega_playground.control.ffpd import FFPD, FFPDConfig
from omega_playground.control.reference import ReferenceTrajectory


def test_ffpd_matches_exported_reference():
    with np.load(Path(__file__).parents[1] / "data/ffpd_reference.npz") as f:
        controller = FFPD(len(f["previous"]), FFPDConfig())
        controller.previous[:] = f["previous"][:, :2]
        got = controller.update(f["reference"], f["reference_velocity"], f["position"], f["velocity"])
        np.testing.assert_allclose(got, f["expected"][:, :2], atol=1e-9, rtol=0)


def test_reference_rounds_corner_and_respects_speed_budget():
    ref = ReferenceTrajectory([[0, 0], [10, 0], [10, 10]], corner_radius=2)
    t = np.arange(0, ref.duration, .1)
    points = np.array([ref.sample(x)[0] for x in t])
    assert np.max(np.linalg.norm(np.diff(points, axis=0), axis=-1) / .1) <= .65 + 1e-6
    assert np.min(np.linalg.norm(points - [10, 0], axis=-1)) > .5
    np.testing.assert_allclose(ref.sample(-1)[0], [0, 0])
    np.testing.assert_allclose(ref.sample(ref.duration + 1)[0], [10, 10])
    assert not ref.xy.flags.writeable


def test_closed_reference_returns_to_start_and_keeps_slot_distances():
    ref = ReferenceTrajectory([[0, 0], [20, 0], [20, 20], [0, 20]], closed=True,
                              slot_offsets=[[0, 0], [-4, 2.4], [-4, -2.4]])
    np.testing.assert_allclose(ref.sample(0)[0], ref.sample(ref.duration)[0])
    for t in np.linspace(0, ref.duration, 20):
        a, b, c = [ref.sample(t, i)[0] for i in range(3)]
        assert np.linalg.norm(b - c) == pytest.approx(4.8)
        assert np.linalg.norm(a - b) == pytest.approx(np.hypot(4, 2.4))


def test_reference_rejects_reversal_and_no_correction_budget():
    with pytest.raises(ValueError):
        ReferenceTrajectory([[0, 0], [10, 0], [0, 0]])
    with pytest.raises(ValueError):
        ReferenceTrajectory([[0, 0], [10, 0]], nominal_speed=1.)


@pytest.mark.parametrize('offset', [.001, .01, .05, .2, .25, .5])
@pytest.mark.parametrize('speed', [.65, 1.5, 3.])
def test_short_live_position_connector_resolves_corner_acceleration(offset, speed):
    # A robot just off the centerline needs a short perpendicular connector.
    # At 20 cm, the old four-sample fillet missed its curvature peak and failed.
    ref = ReferenceTrajectory([[0, -offset], [0, 0], [5, 0]],
                              nominal_speed=speed, max_speed=max(1., speed+.35))
    np.testing.assert_allclose(ref.sample(0)[0], [0, -offset])
    np.testing.assert_allclose(ref.sample(ref.duration)[0], [5, 0])
    # Check the actual 10 Hz reference at different clock phases as well.
    for phase in [0, .025, .05, .075]:
        samples = np.array([ref.sample(t)[0] for t in np.arange(phase, ref.duration+.2, .1)])
        velocity = np.diff(samples, axis=0)/.1
        assert np.max(np.linalg.norm(velocity, axis=-1)) <= speed+1e-6
        assert np.max(np.linalg.norm(np.diff(velocity, axis=0)/.1, axis=-1)) <= 1.001
