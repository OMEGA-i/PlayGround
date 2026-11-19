from types import SimpleNamespace

import pytest

from omega_playground.viewer.cadence import render_rate


def package(hz, physics):
    return SimpleNamespace(physics_hz=physics, layer_schedule=lambda _: SimpleNamespace(interval_s=lambda: 1/hz))


@pytest.mark.parametrize('control,physics,expected', [(50,200,25), (30,120,30), (60,240,30)])
def test_auto_cadence_has_whole_control_and_physics_ticks(control, physics, expected):
    fps = render_rate(package(control, physics))
    assert fps == expected
    assert control / fps == round(control / fps)
    assert physics / fps == round(physics / fps)


def test_override_and_unavailable_frame_generation():
    p = package(50, 200)
    assert render_rate(p, 30) == 30
    for value in (0, -1, float('nan'), float('inf'), 201):
        with pytest.raises(ValueError):
            render_rate(p, value)
    with pytest.raises(ValueError, match='not exposed'):
        render_rate(p, frame_generation='2x')
