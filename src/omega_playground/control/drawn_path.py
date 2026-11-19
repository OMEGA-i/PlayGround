"""Interpolate user-authored Campus XY points without consulting the road graph."""
import numpy as np
from scipy.interpolate import make_interp_spline

from omega_playground.control.mission_player import split_sharp_turns


def drawn_path(keypoints):
    points = np.asarray(keypoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not 2 <= len(points) <= 256 or not np.isfinite(points).all():
        raise ValueError('Provide 2–256 finite XY keypoints')
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if lengths.min() < .02 or lengths.sum() > 2000:
        raise ValueError('Keypoints must be distinct and the stroke shorter than 2000 m')
    legs = []
    for piece in split_sharp_turns(points):
        p = np.asarray(piece)
        knots = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        spline = make_interp_spline(knots, p, k=min(3, len(p)-1))
        samples = np.r_[np.concatenate([np.linspace(a, b, int(np.ceil((b-a)/.25)), endpoint=False)
                                        for a, b in zip(knots[:-1], knots[1:])]), knots[-1]]
        legs.append(spline(samples).tolist())
    route = legs[0] + [point for leg in legs[1:] for point in leg[1:]]
    return dict(route=route, legs=legs, keypoints=points.tolist(), clearance_m=.6,
                length_m=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()))
