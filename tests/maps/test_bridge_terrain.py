import numpy as np
import pytest

from omega_playground.map.terrain import BRIDGE, BRIDGE_COLLISIONS, support_continuous


def test_bridge_steps_are_scoped_and_keep_high_obstacles_closed():
    points = [[27., 157.], [27.05, 157.]]
    assert support_continuous(points, [.25, .15], [BRIDGE])
    assert not support_continuous(points, [.25, .15])
    assert not support_continuous([[0., 0.], [.05, 0.]], [.25, .15], [BRIDGE])
    assert not support_continuous(points, [.33, .85], [BRIDGE])
    assert not support_continuous(points, [.25, np.nan], [BRIDGE])


def test_bridge_slope_handles_grid_quantization_without_changing_flat_limits():
    points = [[45., 157.], [45.05, 157.]]
    assert support_continuous(points, [.7, .95], [BRIDGE], grid_resolution=.5)
    assert not support_continuous(points, [.7, .95], [BRIDGE])
    assert not support_continuous([[0., 0.], [.05, 0.]], [.7, .95], [BRIDGE], grid_resolution=.5)


def test_collision_revision_mismatch_rejects_stale_map(tmp_path):
    from omega_playground.map.navigation import CampusNavigation
    from test_navigation_preview import navigation

    navigation(tmp_path)
    with pytest.raises(ValueError, match='collision_sha256'):
        CampusNavigation(tmp_path, 'scene', 'new-collision')


def test_bridge_end_drop_is_allowed_but_larger_step_is_not():
    points = [[43.2, 157.], [43.205, 157.]]
    assert support_continuous(points, [.02, .195], [BRIDGE])
    assert not support_continuous(points, [.02, .25], [BRIDGE])


def test_bridge_bake_requires_explicit_physics_query():
    from types import SimpleNamespace

    from omega_playground.map.terrain import BRIDGE_COLLISION, bake_bridge
    with pytest.raises(ValueError, match='MJWarp physics asset'):
        bake_bridge(SimpleNamespace(collision_sha256=BRIDGE_COLLISION),
                    np.zeros((2, 2)), np.zeros((2, 2)), [22, 153, 23, 154], .5)


@pytest.mark.parametrize('collision_sha', sorted(BRIDGE_COLLISIONS))
def test_bridge_bake_uses_injected_authored_heights(collision_sha):
    from types import SimpleNamespace

    from omega_playground.map.terrain import BRIDGE_COLLISION, bake_bridge
    calls = []
    def heights(xy, *, allow_missing):
        calls.append((xy.shape, allow_missing))
        return np.full(len(xy), .3)
    support = np.zeros((4, 4))
    clearance = np.zeros((4, 4))
    zones = bake_bridge(SimpleNamespace(collision_sha256=collision_sha), support, clearance,
                        [30, 155, 32, 157], .5, ground=SimpleNamespace(heights=heights))
    assert zones == [BRIDGE]
    assert calls[0][0][1] == 2 and calls[0][1]
    assert np.allclose(support, .3)
    assert np.all(clearance > .6)
