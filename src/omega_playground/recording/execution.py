"""Owned control-boundary snapshots and graceful task cancellation."""
from contextlib import contextmanager
import signal

import numpy as np


class Cancellation:
    def __init__(self):
        self.reason = None

    def request(self, signum, _frame=None):
        self.reason = signal.Signals(signum).name.lower()

    @contextmanager
    def installed(self):
        previous = {s: signal.signal(s, self.request) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            yield self
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)


class PhysicalTrajectory:
    """M controls own M+1 actual states, including the initial and final boundary."""
    def __init__(self, body_names):
        self.body_names = list(body_names)
        self.poses, self.qpos, self.qvel = [], [], []

    def append(self, tick, body_q, qpos, qvel):
        if tick != len(self.poses):
            raise ValueError('Physical trajectory must advance exactly one control tick')
        values = [np.array(a, copy=True) for a in (body_q, qpos, qvel)]
        if values[0].shape != (len(self.body_names), 7) or not all(np.isfinite(a).all() for a in values):
            raise ValueError('Invalid physical snapshot')
        self.poses.append(values[0]); self.qpos.append(values[1]); self.qvel.append(values[2])

    def save(self, output):
        ticks = np.arange(len(self.poses), dtype=np.int64)
        np.savez_compressed(output/'body_states.npz', body_q=self.poses, control_tick=ticks,
            physics_step=ticks*4, sim_time=ticks/50, body_names=self.body_names)
        np.savez_compressed(output/'physics_states.npz', qpos=self.qpos, qvel=self.qvel, physics_step=ticks*4)


def camera_samples(ticks, controls):
    """Only action-row instants are images; a terminal pose is never an extra frame."""
    ticks = np.asarray(ticks)
    expected = np.arange(0, controls, 2)
    indexes = np.flatnonzero((ticks < controls) & (ticks % 2 == 0))
    if not np.array_equal(ticks[indexes], expected):
        raise ValueError('Actual camera snapshot sequence is incomplete')
    return indexes
