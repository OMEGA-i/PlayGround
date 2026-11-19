"""Single-agent keyboard input with an expiring wall-clock lease."""
import time

import torch

from omega_playground.control.locomotion import MODES, validate_locomotion
from omega_playground.policy.base import JoystickBatch


class KeyboardPlayer:
    def __init__(self, mode='walk', speed=.65):
        self.configure(mode, speed)
        self.forward = self.turn = 0
        self.expires = 0.

    def configure(self, mode, speed):
        self.speed = validate_locomotion(mode, speed)
        self.mode_id = MODES[mode]['id']

    def update(self, forward, turn):
        if any(type(v) is not int or v not in (-1, 0, 1) for v in (forward, turn)):
            raise ValueError('Keyboard axes must be -1, 0 or 1')
        self.forward, self.turn = forward, turn
        self.expires = time.monotonic() + .45

    def act(self, view):
        active = time.monotonic() < self.expires
        forward, turn = (self.forward, self.turn) if active else (0, 0)
        q = getattr(view, 'heading_rot', view.root_rot)
        x, y, z, w = q.unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)) + .5*turn
        zero = torch.zeros_like(yaw)
        facing = torch.stack((yaw.cos(), yaw.sin(), zero), -1)
        return JoystickBatch(torch.full_like(yaw, self.mode_id if forward else 2 if turn else 0, dtype=torch.long),
                             zero + (self.speed if forward else 0 if turn else -1),
                             facing * forward, facing, zero, zero-1)
