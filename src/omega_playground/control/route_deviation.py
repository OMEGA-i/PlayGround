"""Spatial route deviation, independent of the reference clock and speed."""
import numpy as np


class LocalRouteProjection:
    """Follow a contiguous arc window, avoiding shortcuts to another crossing/lap.

    The cursor starts at the active leg's beginning and follows measured progress.
    Large off-route excursions do not move the cursor. A new task/leg resets it.
    """
    def __init__(self, reference, role):
        forward, left = reference.offsets[role]
        c, s = np.cos(reference.yaw), np.sin(reference.yaw)
        self.xy = reference.xy + np.c_[forward*c-left*s, forward*s+left*c]
        self.delta = np.diff(self.xy, axis=0)
        self.length2 = np.sum(self.delta**2, axis=1)
        self.arc = np.r_[0., np.cumsum(np.sqrt(self.length2))]
        self.reference_s = reference.s
        self.progress = 0.
        self.reference_distance = 0.
        self.tangent = np.array([np.cos(reference.yaw[0]), np.sin(reference.yaw[0])])

    def project(self, position):
        # 10 m each way is local geometry, not a time/speed-based search window.
        lo = max(0, np.searchsorted(self.arc, self.progress-10.)-1)
        hi = min(len(self.delta), np.searchsorted(self.arc, self.progress+10.)+1)
        delta = self.delta[lo:hi]
        fraction = np.clip(np.sum((position-self.xy[lo:hi])*delta, axis=1)
                           / np.maximum(self.length2[lo:hi], 1e-12), 0., 1.)
        projected = self.xy[lo:hi] + fraction[:, None]*delta
        distances = np.sum((projected-position)**2, axis=1)
        index = int(np.argmin(distances))
        if distances[index] < 4.:
            self.progress = self.arc[lo+index] + fraction[index]*np.sqrt(self.length2[lo+index])
        i = lo+index
        if self.length2[i] > 1e-12:
            self.tangent = self.delta[i]/np.sqrt(self.length2[i])
        distance = self.reference_s[i] + fraction[index]*(self.reference_s[i+1]-self.reference_s[i])
        self.reference_distance = float(distance)
        return projected[index]-position, self.reference_distance

    def error(self, position):
        return self.project(position)[0]
