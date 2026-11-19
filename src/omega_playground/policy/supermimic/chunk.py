"""Physics-clock boundaries for execute-then-plan SuperMimic inference."""
from collections import deque

import numpy as np
import torch

CONTROL_HZ = 50
PHYSICS_HZ = 200
LOOKAHEAD = 10


class ProprioHistory:
    """Owned physical states; the planner receives four samples at 25 FPS."""
    def __init__(self, state, tick):
        self.rows = deque(maxlen=7)
        self.add(state, tick)

    def add(self, state, tick):
        if self.rows and tick != self.rows[-1][0] + 1:
            raise ValueError('Proprio history must advance by one physics control tick')
        self.rows.append((tick, state.detach().cpu().numpy()[0].copy()))

    def snapshot(self):
        tick = self.rows[-1][0]
        rows = dict(self.rows)
        first_tick, first = self.rows[0]
        wanted = np.arange(tick-6, tick+1, 2, dtype=np.int64)
        actual = np.maximum(wanted, first_tick)
        return dict(states=np.stack([rows.get(int(t), first) for t in wanted]),
                    ticks=actual, valid=wanted >= first_tick,
                    physics_steps=actual*4, timestamp=actual/CONTROL_HZ)


def streaming_window(queue, joints):
    """Current + ten future samples; only missing lookahead uses terminal hold."""
    if queue is None or len(queue) < 2:
        raise RuntimeError('Chunk is exhausted; plan from the new physical state')
    count = min(len(queue), LOOKAHEAD+1)
    window = queue[:count].clone()
    if count < LOOKAHEAD+1:
        padding = queue[-1:].repeat(LOOKAHEAD+1-count, 1)
        padding[:, 7:13] = 0  # reference root linear/angular velocity
        padding[:, 13+joints:13+2*joints] = 0
        window = torch.cat((window, padding))
    valid = torch.arange(LOOKAHEAD+1, device=queue.device) < count
    return window, valid
