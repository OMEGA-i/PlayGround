"""Show and validate the same collision-derived clearance used by the Player."""
from io import BytesIO

import numpy as np
from PIL import Image

from omega_playground.map.navigation import resample

RADIUS = .6


def inspect_path(navigation, route, radius=RADIUS):
    samples = np.asarray(resample(route, min(.1, navigation.resolution * .4)))
    valid = navigation.valid(samples, radius)
    invalid = samples[~valid]
    message = ''
    if len(invalid):
        x, y = invalid[0]
        message = f'轨迹在 ({x:.2f}, {y:.2f}) 进入障碍、无支撑区域或不足 {radius:.1f} m 净空；请调整关键点'
    return {'valid': bool(valid.all()), 'clearance_m': radius,
            'invalid_points': invalid.tolist(), 'message': message}


def occupancy_png(navigation):
    with np.load(navigation.directory / 'layers.npz') as layers:
        blocked = layers['collision_blocked'] | ~np.isfinite(layers['support_height'])
    rgba = np.zeros((*navigation.shape, 4), dtype=np.uint8)
    rgba[navigation.clearance < RADIUS] = [245, 170, 55, 100]
    rgba[blocked] = [244, 78, 78, 135]
    result = BytesIO()
    Image.fromarray(rgba[::-1]).save(result, format='PNG')
    return result.getvalue()
