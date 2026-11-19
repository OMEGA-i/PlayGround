"""Bounded, collision-ray terrain baking for the surveyed raised Campus bridge."""
import numpy as np

BRIDGE_COLLISION = '0ebcad5c37b0dea4bec55e952d6b7815d59b90e2bf76cba5e5e217a44c810649'
# r11 removes only fully contained road boxes. The retained bridge geometry
# and all its attributes match r10; see campus_r11_geometry_audit.json.
BRIDGE_COLLISIONS = frozenset((BRIDGE_COLLISION,
    '5038be3278df6d5c0926cf609611fb1208a22165699bd7d200ed7c443e326d21',
    # r12 entrance candidate: all r11 prim metadata, attributes and relationships
    # unchanged; only three matching visible/collision entrance wedges added.
    # This enables terrain sampling, not locomotion acceptance.
    'c07e2123143d521fedc1a235e29db5f67a4f09fce7a04811a8fc7b273da145f6',
    # r11 hull coordinates normalized; audited world error <= 5.401e-6 m.
    '2b8a845178caacabca0e2c7a20a0399080fcd824b192149662431546bc71e76b'))
BRIDGE = {'bounds': [22., 153., 61., 161.], 'step_limit_m': .20,
          'slope_limit': .4, 'max_height_m': 1.5, 'resolution': .05}


def support_continuous(points, heights, zones=(), *, grid_resolution=0.):
    points, heights = np.asarray(points), np.asarray(heights)
    if not np.isfinite(heights).all():
        return False
    limit = np.full(max(0, len(heights)-1), .05001)
    for zone in zones:
        bounds = np.asarray(zone['bounds'])
        inside = ((points >= bounds[:2]) & (points <= bounds[2:])).all(axis=1)
        distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
        # A nearest-cell height query can cross a full grid cell in a tiny XY step.
        allowance = zone['step_limit_m'] + zone['slope_limit'] * np.maximum(
            distance, grid_resolution * np.sqrt(2))
        limit = np.where(inside[:-1] & inside[1:], allowance, limit)
    return bool((np.abs(np.diff(heights)) <= limit).all())


def bake_bridge(scene, support, clearance, bounds, resolution, *, ground=None):
    """Use the actual highest collider surface, retaining rails/trees/steps as obstacles.

    This explicit revision-bound experiment permits <=20 cm approach steps, not
    arbitrary elevated roofs. It grants no locomotion acceptance.
    """
    if scene.collision_sha256 not in BRIDGE_COLLISIONS:
        return []
    from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

    if ground is None:
        raise ValueError('Bridge baking requires a prepared MJWarp physics asset for terrain sampling')

    zone = dict(BRIDGE)
    lo_x, lo_y, hi_x, hi_y = zone['bounds']
    r = zone['resolution']
    # Margin keeps the sampling rectangle itself from introducing artificial walls.
    xs = np.arange(lo_x-2+r/2, hi_x+2, r)
    ys = np.arange(lo_y-2+r/2, hi_y+2, r)
    xx, yy = np.meshgrid(xs, ys)
    heights = ground.heights(np.c_[xx.ravel(), yy.ravel()], allow_missing=True).reshape(xx.shape)
    finite = np.isfinite(heights)
    h = np.where(finite, heights, 100.)
    jumps = maximum_filter(h, 3) - minimum_filter(h, 3)
    free = finite & (h >= -.15) & (h <= zone['max_height_m'])
    free &= jumps <= zone['step_limit_m'] + zone['slope_limit'] * 2*r + 1e-5
    distances = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1]*r-r*.71
    gy, gx = np.indices(support.shape)
    wx, wy = bounds[0]+(gx+.5)*resolution, bounds[1]+(gy+.5)*resolution
    use = (wx >= lo_x) & (wx <= hi_x) & (wy >= lo_y) & (wy <= hi_y)
    ix = np.rint((wx[use]-xs[0])/r).astype(int)
    iy = np.rint((wy[use]-ys[0])/r).astype(int)
    support[use] = heights[iy, ix]
    # Deduct the coarse cell diagonal: every accepted cell retains body clearance.
    clearance[use] = distances[iy, ix] - resolution*np.sqrt(2)/2
    return [zone]
