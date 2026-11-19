import numpy as np

from omega_playground.viewer.formation_camera import FormationCamera, visible_roots


def test_wall_blocks_oblique_view_but_camera_finds_clear_view():
    roots = np.array([[0., 0, 1], [12, 0, 1], [24, 0, 1]])
    target = roots.mean(axis=0)
    boxes = np.array([[[-30, -30, 0], [5, -3, 12]]], dtype=float)
    camera = FormationCamera(boxes)
    assert visible_roots(target+30*np.array([-.866, -.5, .6]), roots, boxes) < 3
    eye = camera.eye(roots, target, 30)
    assert visible_roots(eye, roots, boxes) == 3
    assert camera.choice != 0
    np.testing.assert_allclose(np.linalg.norm(eye-target), 30*np.linalg.norm([-.866, -.5, .6]))
    np.testing.assert_allclose(camera.eye(roots+.01, target+.01, 30), eye+.01)


def test_parallel_rays_and_eye_inside_obstacle():
    roots = np.array([[0., 0, 1], [1, 0, 1]])
    boxes = np.array([[[-1, -1, 2], [1, 1, 3]]], dtype=float)
    assert visible_roots(np.array([0., 0, 4]), roots, boxes) == 0
    assert visible_roots(np.array([0., 0, 2.5]), roots, boxes) == 0
    assert visible_roots(np.array([3., 0, 1]), roots, boxes) == 2


def test_nonconvex_terrain_bounds_containing_target_are_inconclusive():
    roots = np.array([[0., 0, 1]])
    boxes = np.array([[[-100., -100, -2], [100, 100, 4]]])
    assert visible_roots(np.array([0., 0, 30]), roots, boxes) == 1
