"""Conservative static-scene visibility for automatic formation cameras."""
import math

import numpy as np

from omega_playground.viewer.affine import rotation, static_mesh_affines


def scene_bounds(model, instances=()):
    """World AABBs of visible static geoms, including authored affine instances."""
    overrides = {}
    if instances:
        ids, positions, matrices, _ = static_mesh_affines(model, instances)
        overrides = dict(zip(ids, zip(positions, matrices, strict=True), strict=True))
    boxes = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] != 0 or model.geom_group[i] > 2 or model.geom_rgba[i, 3] <= 0:
            continue
        pos, matrix = overrides.get(i, (model.geom_pos[i], rotation(model.geom_quat[i])))
        center, half = model.geom_aabb[i, :3], model.geom_aabb[i, 3:]
        center, half = pos+matrix@center, np.abs(matrix)@half
        # Infinite planes and degenerate surfaces cannot enclose the camera.
        if np.isfinite(half).all() and np.max(half) < 1e8:
            boxes.append([center-half, center+half])
    return np.asarray(boxes, dtype=float).reshape(-1, 2, 3)


def visible_roots(eye, roots, boxes):
    if not len(boxes):
        return len(roots)
    roots = np.asarray(roots)
    lower = np.minimum(roots.min(axis=0), eye)
    upper = np.maximum(roots.max(axis=0), eye)
    boxes = boxes[np.all(boxes[:, 1] >= lower, axis=1) & np.all(boxes[:, 0] <= upper, axis=1)]
    if not len(boxes):
        return len(roots)
    delta = roots-eye
    parallel = np.abs(delta) < 1e-10
    safe = np.where(parallel, 1., delta)
    times = (boxes[None, :, :, :]-eye)/safe[:, None, None, :]
    lower, upper = times.min(axis=2), times.max(axis=2)
    outside = (eye < boxes[:, 0]) | (eye > boxes[:, 1])
    lower = np.where(parallel[:, None, :], -np.inf, lower)
    upper = np.where(parallel[:, None, :], np.inf, upper)
    enter, leave = lower.max(axis=-1), upper.min(axis=-1)
    blocked = (np.maximum(enter, 0) < np.minimum(leave, .98)) & ~np.any(parallel[:, None, :] & outside, axis=-1)
    # A nonconvex terrain mesh may wrap the target in its AABB despite a clear
    # surface. Such a box cannot establish occlusion without triangle queries.
    target_inside = np.all((roots[:, None] >= boxes[None, :, 0]) &
                           (roots[:, None] <= boxes[None, :, 1]), axis=-1)
    blocked &= ~target_inside
    return int(np.count_nonzero(~blocked.any(axis=1)))


class FormationCamera:
    def __init__(self, boxes):
        self.boxes = boxes
        self.choice = 0
        self.previous = None

    def eye(self, roots, target, distance):
        # Keep the familiar oblique view first; raise/orbit only when occluded.
        directions = [np.array([-.866, -.5, .6])]
        for height in (1.2, 2.4, 4.):
            for angle in (210, 120, 300, 30):
                a = math.radians(angle)
                directions.append(np.array([math.cos(a), math.sin(a), height]))
        directions.append(np.array([.001, 0., 4.]))
        norm = np.linalg.norm(directions[0])
        directions = [d*norm/np.linalg.norm(d) for d in directions]
        if self.previous is None or self.previous.shape != roots.shape or np.max(np.abs(roots-self.previous)) > .25:
            best = visible_roots(target+distance*directions[self.choice], roots, self.boxes)
            # Retain the previous view on equal scores; most updates need one query.
            for index, direction in enumerate(directions):
                if best == len(roots):
                    break
                if index == self.choice:
                    continue
                score = visible_roots(target+distance*direction, roots, self.boxes)
                if score > best:
                    self.choice, best = index, score
            self.previous = roots.copy()
        return target+distance*directions[self.choice]
