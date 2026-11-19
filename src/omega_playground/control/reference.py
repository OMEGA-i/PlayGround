"""Rounded, time-parameterized route references; never advanced by robot progress."""

import math
from copy import copy

import numpy as np


class ReferenceTrajectory:
    def __init__(self, route, *, closed=False, nominal_speed=0.65, corner_radius=3.,
                 max_speed=1., max_acceleration=1., margin=0.1, slot_offsets=((0., 0.),), spacing=0.05, preserve_path=False):
        points = np.asarray(route, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Route must contain finite XY points")
        if closed and len(points) > 1 and np.linalg.norm(points[0] - points[-1]) < 1e-8:
            points = points[:-1]
        if len(points) < (3 if closed else 2) or np.any(np.linalg.norm(np.diff(points, axis=0), axis=-1) < 1e-6):
            raise ValueError("Route has too few or duplicate points")
        # Grid planners may sample long straight segments every metre. Those samples
        # must not artificially cap the fillet radius at the neighboring sample spacing.
        keep = []
        for i in range(len(points)):
            if not closed and i in (0, len(points) - 1):
                keep.append(i)
                continue
            u, v = points[i] - points[i - 1], points[(i + 1) % len(points)] - points[i]
            if abs(u[0] * v[1] - u[1] * v[0]) > 1e-8 or np.dot(u, v) <= 0:
                keep.append(i)
        points = points[keep]
        if not 0 < nominal_speed <= max_speed - margin or corner_radius <= 0 or max_acceleration <= 0:
            raise ValueError("Reference requires positive limits and speed below the correction budget")
        self.closed, self.nominal_speed, self.corner_radius = closed, nominal_speed, corner_radius
        self.offsets = np.asarray(slot_offsets, dtype=np.float64)
        if self.offsets.ndim != 2 or self.offsets.shape[1] != 2 or not np.isfinite(self.offsets).all():
            raise ValueError("Slot offsets must be finite (forward, left) pairs")
        entry, exit = points.copy(), points.copy()
        corners = {}
        for i in range(0 if preserve_path else len(points)):
            if not closed and i in (0, len(points) - 1):
                continue
            before, after = points[i] - points[i - 1], points[(i + 1) % len(points)] - points[i]
            length0, length1 = np.linalg.norm(before), np.linalg.norm(after)
            u, v = before / length0, after / length1
            turn = math.atan2(u[0] * v[1] - u[1] * v[0], np.dot(u, v))
            if abs(turn) < 1e-7:
                continue
            if abs(turn) > math.pi - 1e-3:
                raise ValueError("A route reversal needs an explicit turning loop")
            tangent = math.tan(abs(turn) / 2)
            d = min(corner_radius * tangent, .45 * length0, .45 * length1)
            entry[i], exit[i] = points[i] - u * d, points[i] + v * d
            corners[i] = u, v, d
        path, yaw, curvature = [], [], []
        for i in range(len(points) if closed else len(points) - 1):
            j = (i + 1) % len(points)
            start, end = exit[i], entry[j]
            vector = end - start
            count = max(2 if preserve_path else 1, math.ceil(np.linalg.norm(vector) / spacing))
            heading = math.atan2(vector[1], vector[0])
            for t in np.arange(count) / count:
                path.append(start + t * vector)
                yaw.append(heading)
                curvature.append(0.)
            if j in corners:
                u, v, d = corners[j]
                # Quintic Bezier: collinear, equally spaced end controls give zero
                # endpoint curvature, so rigid offset slots have continuous velocity.
                controls = np.array([entry[j], entry[j] + .4 * d * u, entry[j] + .8 * d * u,
                                     exit[j] - .8 * d * v, exit[j] - .4 * d * v, exit[j]])
                # Short live-position connectors still need enough samples to resolve
                # the curvature peak; spatial spacing alone can miss it and
                # produce a time profile that fails its acceleration check.
                count = max(32, math.ceil(2 * d / spacing))
                for t in np.arange(count) / count:
                    def blend(values, t=t):
                        degree = len(values) - 1
                        weights = [math.comb(degree, k) * t ** k * (1 - t) ** (degree - k)
                                   for k in range(degree + 1)]
                        return np.asarray(weights) @ values

                    derivative = blend(5 * np.diff(controls, axis=0))
                    second = blend(20 * np.diff(controls, n=2, axis=0))
                    path.append(blend(controls))
                    yaw.append(math.atan2(derivative[1], derivative[0]))
                    curvature.append(abs(derivative[0] * second[1] - derivative[1] * second[0])
                                     / np.linalg.norm(derivative) ** 3)
        path.append(exit[0] if closed else points[-1])
        yaw.append(yaw[0] if closed else yaw[-1])
        curvature.append(curvature[0] if closed else 0.)
        self.xy = np.asarray(path)
        self.yaw = np.unwrap(yaw)
        self.s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=-1))]
        self.length = float(self.s[-1])
        self.curvature = np.asarray(curvature)
        if preserve_path:
            tangent = np.gradient(self.xy, self.s, axis=0)
            self.yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
            self.curvature = np.abs(np.gradient(self.yaw, self.s))
        self._parameterize(max_speed, max_acceleration, margin)
        for array in (self.xy, self.yaw, self.s, self.speed, self.times, self.offsets):
            array.setflags(write=False)
        samples = np.arange(0., self.duration + .1, .1)
        for role in range(len(self.offsets)):
            xy = np.array([self.sample(t, role)[0] for t in samples])
            velocity = np.diff(xy, axis=0) / .1
            acceleration = np.diff(velocity, axis=0) / .1
            if np.linalg.norm(velocity, axis=-1).max() > max_speed - margin + 1e-5:
                raise ValueError(f"Reference slot {role} exceeds its speed budget")
            if np.linalg.norm(acceleration, axis=-1).max() > max_acceleration + 1e-3:
                raise ValueError(f"Reference slot {role} exceeds its acceleration budget")

    def _parameterize(self, max_speed, max_acceleration, margin=.1):
        kappa = self.curvature
        outer = 1 + np.linalg.norm(self.offsets, axis=-1).max() * kappa
        # Reserve acceleration for tangent speed changes in addition to centripetal acceleration.
        speed = np.minimum(self.nominal_speed, (max_speed - margin) / outer)
        speed = np.minimum(speed, np.sqrt(.5 * max_acceleration / np.maximum(kappa * outer, 1e-12)))
        speed[[0, -1]] = 0.
        tangent_accel = .5 * max_acceleration / outer.max()
        for i in range(1, len(speed)):
            speed[i] = min(speed[i], math.sqrt(speed[i - 1] ** 2 + 2 * tangent_accel * (self.s[i] - self.s[i - 1])))
        for i in range(len(speed) - 2, -1, -1):
            speed[i] = min(speed[i], math.sqrt(speed[i + 1] ** 2 + 2 * tangent_accel * (self.s[i + 1] - self.s[i])))
        self.speed = speed
        self.times = np.r_[0., np.cumsum(2 * np.diff(self.s) / (speed[:-1] + speed[1:]))]
        self.duration = float(self.times[-1])

    def retimed(self, nominal_speed, max_speed, max_acceleration):
        """Change timing on identical geometry; curvature still limits corner speed."""
        result = copy(self)
        result.nominal_speed = nominal_speed
        result._parameterize(max_speed, max_acceleration)
        result.speed.setflags(write=False)
        result.times.setflags(write=False)
        return result

    def distance_at(self, t):
        t = float(np.clip(t, 0., self.duration))
        i = min(max(0, int(np.searchsorted(self.times, t, side='right'))-1), len(self.times)-2)
        tau = t-self.times[i]
        acceleration = (self.speed[i+1]-self.speed[i])/(self.times[i+1]-self.times[i])
        return self.s[i]+self.speed[i]*tau+.5*acceleration*tau*tau

    def time_at_distance(self, distance):
        distance = float(np.clip(distance, 0., self.length))
        i = min(max(0, int(np.searchsorted(self.s, distance, side='right'))-1), len(self.s)-2)
        ds = distance-self.s[i]
        acceleration = (self.speed[i+1]-self.speed[i])/(self.times[i+1]-self.times[i])
        end_speed = math.sqrt(max(0., self.speed[i]**2+2*acceleration*ds))
        tau = 2*ds/max(self.speed[i]+end_speed, 1e-12)
        return float(self.times[i]+tau)

    def sample(self, t, role=0):
        t = float(np.clip(t, 0., self.duration))
        i = min(int(np.searchsorted(self.times, t, side="right")) - 1, len(self.times) - 2)
        dt = self.times[i + 1] - self.times[i]
        tau = t - self.times[i]
        acceleration = (self.speed[i + 1] - self.speed[i]) / dt
        distance = self.s[i] + self.speed[i] * tau + .5 * acceleration * tau ** 2
        xy = np.array([np.interp(distance, self.s, self.xy[:, i]) for i in range(2)])
        yaw = float(np.interp(distance, self.s, self.yaw))
        forward, left = self.offsets[role]
        return xy + [forward * math.cos(yaw) - left * math.sin(yaw),
                     forward * math.sin(yaw) + left * math.cos(yaw)], yaw

    def describe(self):
        return {"schema": "omega.reference/1", "length_m": self.length, "duration_s": self.duration,
                    "nominal_speed": self.nominal_speed, "corner_radius": self.corner_radius,
                    "slot_offsets": self.offsets.tolist(), "closed": self.closed}
