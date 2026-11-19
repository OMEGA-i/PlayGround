"""Float64, per-agent feedforward PD velocity commands (contracts §6.1)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FFPDConfig:
    kp: float = 1.0
    kd: float = 0.3
    max_speed: float = 1.0
    max_acceleration: float = 1.0
    dt: float = 0.1
    lateral_kp: float = 1.5
    lateral_kd: float = 0.3
    lateral_max_speed: float = 0.4

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in
                   (self.kp, self.kd, self.max_speed, self.max_acceleration, self.dt,
                    self.lateral_kp, self.lateral_kd, self.lateral_max_speed)):
            raise ValueError("FFPD parameters must be finite and positive")


def limit(v, maximum):
    return v * np.minimum(1., maximum / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


class FFPD:
    def __init__(self, count: int, config: FFPDConfig):
        self.config = config
        self.previous = np.zeros((count, 2), dtype=np.float64)
        self.report = {}

    def reset(self, ids, velocity):
        self.previous[ids] = limit(np.asarray(velocity, dtype=np.float64)[..., :2], self.config.max_speed)

    def update(self, reference, reference_velocity, position, velocity, *, route_error=None, route_tangent=None,
               route_progress_error=None):
        rp, rv, p, v = [np.asarray(a, dtype=np.float64)[..., :2]
                        for a in (reference, reference_velocity, position, velocity)]
        if any(a.shape != self.previous.shape or not np.isfinite(a).all() for a in (rp, rv, p, v)):
            raise ValueError("FFPD expects finite per-agent XY states")
        c = self.config
        raw = rv + c.kp * (rp - p) + c.kd * (rv - v)
        desired = limit(raw, c.max_speed)
        lateral_report = {}
        if route_error is not None or route_tangent is not None:
            error, tangent = [np.asarray(a, dtype=np.float64) for a in (route_error, route_tangent)]
            if any(a.shape != p.shape or not np.isfinite(a).all() for a in (error, tangent)):
                raise ValueError("Route feedback requires finite XY error and unit tangents")
            if not np.allclose(np.linalg.norm(tangent, axis=-1), 1., atol=1e-6):
                raise ValueError("Route tangents must be unit vectors")
            normal = np.c_[-tangent[:, 1], tangent[:, 0]]
            cross_error = np.sum(error*normal, axis=-1)
            cross_velocity = np.sum(v*normal, axis=-1)
            # No deadband: correct centimetre-scale offsets and damp outward motion.
            cross = np.clip(c.lateral_kp*cross_error-c.lateral_kd*cross_velocity,
                            -min(c.lateral_max_speed, c.max_speed), min(c.lateral_max_speed, c.max_speed))
            along_budget = np.sqrt(np.maximum(0., c.max_speed**2-cross**2))
            along_raw = np.sum(raw*tangent, axis=-1)
            if route_progress_error is not None:
                progress_error = np.asarray(route_progress_error, dtype=np.float64)
                if progress_error.shape != (len(p),) or not np.isfinite(progress_error).all():
                    raise ValueError("Route progress error must be finite per agent")
                feedforward = np.linalg.norm(rv, axis=-1)
                # Arc progress remains meaningful after the timed reference has
                # turned a corner ahead of the robot's local tangent.
                along_raw = feedforward + c.kp*progress_error + c.kd*(feedforward-np.sum(v*tangent, axis=-1))
            along = np.clip(along_raw, -along_budget, along_budget)
            desired = along[:, None]*tangent + cross[:, None]*normal
            lateral_report = {"lateral_error": cross_error, "lateral_velocity": cross_velocity,
                              "lateral_desired_speed": cross}

        delta = desired - self.previous
        self.previous += limit(delta, c.max_acceleration * c.dt)
        self.report = {**lateral_report, "position_error": rp - p, "velocity_error": rv - v,
                           "speed_saturated": np.linalg.norm(raw, axis=-1) > c.max_speed,
                           "acceleration_saturated": np.linalg.norm(delta, axis=-1) > c.max_acceleration * c.dt}
        return self.previous.copy()
