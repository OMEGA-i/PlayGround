"""Choose recording cadence from the policy's low-level control schedule."""
import math


def render_rate(package, requested=None, frame_generation='off'):
    if frame_generation != 'off':
        raise ValueError('DLSS 2x frame generation is unavailable in the current offscreen video pipeline; '
                         'generated frames are not exposed to the encoder. Use --dlss-frame-generation off.')
    if requested is None:
        control = 1. / package.layer_schedule('wbc').interval_s()
        # 50 Hz -> 25 FPS, 30 Hz -> 30 FPS; other policies use a whole number of control ticks.
        requested = control / max(1, math.ceil(control / 30. - 1e-9))
    if not math.isfinite(requested) or not 0 < requested <= package.physics_hz:
        raise ValueError('render-hz must be finite, positive and no greater than physics rate')
    return float(requested)
