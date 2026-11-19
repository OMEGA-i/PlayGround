"""SONIC gait/speed settings; route and FFPD own movement and facing."""
import math

MODES = {'walk': {'id': 2, 'max_speed': 2.}, 'run': {'id': 3, 'max_speed': 3.}}


def validate_locomotion(mode, speed):
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError('SONIC mode must be walk or run')
    if isinstance(speed, bool):
        raise TypeError('Speed must be a finite number')
    speed = float(speed)
    if not math.isfinite(speed) or not .1 <= speed <= MODES[mode]['max_speed']:
        raise ValueError(f"{mode} speed must be within 0.1–{MODES[mode]['max_speed']:g} m/s")
    return speed
