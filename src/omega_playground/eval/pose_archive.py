"""Bounded asynchronous device-to-host capture for diagnostic pose replay."""
from collections import deque

import torch


class DevicePoseArchive:
    def __init__(self, device, capacity=2):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.stream = torch.cuda.Stream(device=device)
        self.capacity = capacity
        self.pending = deque()
        self.frames = []
        self.times = []
        self.free = []

    def _drain_one(self):
        host, event = self.pending.popleft()
        event.synchronize()
        self.frames.append(host.numpy().copy())
        self.free.append(host)

    def append(self, poses, sim_time):
        if poses.device != self.stream.device:
            raise ValueError('pose archive requires tensors on its CUDA device')
        if len(self.pending) >= self.capacity:
            self._drain_one()
        host = self.free.pop() if self.free else torch.empty_like(poses, device='cpu', pin_memory=True)
        if host.shape != poses.shape or host.dtype != poses.dtype:
            raise ValueError('pose archive layout changed')
        self.stream.wait_stream(torch.cuda.current_stream(poses.device))
        with torch.cuda.stream(self.stream):
            host.copy_(poses, non_blocking=True)
            event = self.stream.record_event()
            poses.record_stream(self.stream)
        self.pending.append((host, event))
        self.times.append(sim_time)

    def finish(self):
        while self.pending:
            self._drain_one()
        self.free.clear()
        return self.frames, self.times
