"""Independent route clocks and controller state in one shared physical world."""

from dataclasses import fields

import numpy as np
import torch

from omega_playground.control.player_factory import device_player_enabled, route_player
from omega_playground.policy.base import JoystickBatch, RobotBatchView


def subset(view, ids):
    # Teams and leaders use contiguous IDs. Basic slices avoid a CUDA gather
    # kernel and host-index upload for every RobotBatchView field.
    if isinstance(ids, (list, tuple, np.ndarray)) and len(ids):
        indices = np.asarray(ids)
        if indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer) and (np.diff(indices) == 1).all():
            ids = slice(int(indices[0]), int(indices[-1]) + 1)
    return RobotBatchView(**{f.name: getattr(view, f.name)[ids] if isinstance(getattr(view, f.name), torch.Tensor) else getattr(view, f.name)
                             for f in fields(RobotBatchView)})


def host_view(view):
    """One transfer for CPU route controllers instead of per-team scalar reads."""
    if view.body_pos.device.type == 'cpu':
        return view
    names = [f.name for f in fields(RobotBatchView) if isinstance(getattr(view, f.name), torch.Tensor)]
    values = [getattr(view, name) for name in names]
    packet = torch.cat([value.reshape(-1) for value in values]).cpu()
    result, offset = {}, 0
    for name, value in zip(names, values, strict=True):
        count = value.numel()
        result[name] = packet[offset:offset+count].reshape(value.shape).to(value.dtype)
        offset += count
    return RobotBatchView(**result, anchor=view.anchor, heading_offset=view.heading_offset)


class TaskControllerBatch:
    name = 'task_controllers'
    # Batching adapter, not a Player: each entry is one NPC or one Squad Master.

    def __init__(self, specs, device, navigation, family="sonic"):
        self.players, self.ids = [], []
        self.device = device
        offset = 0
        for spec in specs:
            n = len(spec.get('reference', {}).get('slot_offsets', [[0, 0]]))
            player_device = device if n == 1 and device_player_enabled(device) else 'cpu'
            player = route_player(family, spec, n, player_device)
            player.navigation = navigation
            self.players.append(player)
            self.ids.append(np.arange(offset, offset + n))
            offset += n
        self.num_agents = offset
        self.schedule = self.players[0].schedule
        if any(p.schedule != self.schedule for p in self.players):
            raise ValueError('All teams must share a Player schedule')
        self.hold = None

    def reset(self, agent_ids, view):
        selected = set(agent_ids.tolist())
        for player, ids in zip(self.players, self.ids, strict=True):
            local = [i for i, agent in enumerate(ids) if agent in selected]
            if local:
                player.reset(torch.tensor(local, device=view.root_pos.device), subset(view, ids))

    def act(self, view, task, sim_time):
        # Host-only Players share one diagnostic transfer. Device Players must
        # retain the original physics tensors, including after task replacement.
        cpu_view = host_view(view) if any(torch.device(p.device).type == 'cpu' for p in self.players) else None
        commands = [p.act(subset(cpu_view if torch.device(p.device).type == 'cpu' else view, ids), task, sim_time)
                    for p, ids in zip(self.players, self.ids, strict=True)]
        return JoystickBatch(**{f.name: torch.cat([getattr(c, f.name).to(self.device) for c in commands])
                                for f in fields(JoystickBatch)})

    def check(self, view, sim_time):
        view = host_view(view)
        reports = []
        for team, (player, ids) in enumerate(zip(self.players, self.ids, strict=True)):
            reports.append(player.check(subset(view, ids), sim_time))
            if player.hold is not None and self.hold is None:
                self.hold = {**player.hold, 'team': team,
                             'agents': ids[player.hold['agents']].tolist()}
        return {k: np.concatenate([r[k] for r in reports]) for k in reports[0]}
