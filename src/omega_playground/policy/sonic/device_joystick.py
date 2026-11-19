"""Batched G1 Joystick state. Pose buffers and decisions stay on the device.

The runner may compact the replan mask to schedule inference. No command,
pose, frame count or history is converted to Python inside this adapter.
"""
import math
from dataclasses import fields

import numpy as np
import torch

from omega_playground.policy.base import JoystickBatch, MotionReference

from . import torch_motion


def gather(q, indices):
    return q[torch.arange(len(q), device=q.device)[:, None], indices]


def sample(q, times, fps, length):
    # Fixed capacity with variable valid lengths. Clamp to the authored endpoint,
    # then reuse the release time-grid and quaternion convention.
    end = torch_motion._time_grid(q.shape[1], fps, q.device)[length-1]
    return torch_motion.sample_qpos(q, times.minimum(end[:, None]), fps)


class DeviceJoystick:
    def __init__(self, owner):
        self.owner = owner
        self.device = torch.device(owner.device)
        n, dev = owner.num_agents, self.device
        self.capacity = math.floor(owner.planner.max_frames/owner.planner_hz*owner.hz)+owner.lookahead
        self.q = torch.zeros(n, self.capacity, 36, dtype=torch.float64, device=dev)
        self.q[..., 3] = 1
        self.v = torch.zeros(n, self.capacity, 29, dtype=torch.float64, device=dev)
        self.length = torch.ones(n, dtype=torch.long, device=dev)
        self.start = torch.zeros(n, dtype=torch.long, device=dev)
        self.valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.polls = torch.zeros(n, dtype=torch.long, device=dev)
        self.last = None
        self.to_planner = torch.as_tensor(owner.to_planner, device=dev)
        self.to_package = torch.as_tensor(owner.to_package, device=dev)
        self.root_xyz = torch.as_tensor(owner.root_xyz, device=dev)
        c = owner.initial
        self.initial = JoystickBatch(torch.full((n,), c.mode, dtype=torch.long, device=dev),
            torch.full((n,), c.speed, dtype=torch.float64, device=dev),
            torch.tensor(c.movement, device=dev).double().expand(n, -1),
            torch.tensor(c.facing, device=dev).double().expand(n, -1),
            torch.zeros(n, device=dev), torch.full((n,), c.height, dtype=torch.float64, device=dev))
        self.last = self.initial
        self.indices = torch.arange(self.capacity, device=dev)[None]
        self.context_times = torch.tensor(np.stack([
            (cursor+owner.lookahead)/owner.hz + np.arange(owner.context_frames)/owner.planner_hz
            for cursor in range(self.capacity+1)]), device=dev)
        self.resampled_lengths = torch.tensor([math.floor(k/owner.planner_hz*owner.hz)
            for k in range(owner.planner.max_frames+1)], device=dev)

    def reset(self, ids):
        self.valid[ids] = False
        self.polls[ids] = 0

    def should_replan(self, commands, tick):
        o = self.owner
        c = self.on_device(commands)
        if tick == 0 or tick % o.poll_ticks:
            return ~self.valid
        polls = self.polls + self.valid.long()
        threshold = torch.full_like(polls, o.poll_count['default'])
        for key, value in o.poll_count.items():
            if key != 'default':
                threshold = torch.where(c.mode == int(key), value, threshold)
        timed = polls >= threshold
        self.polls = torch.where(timed, 0, polls)
        changed = (c.mode != self.last.mode) | (c.facing != self.last.facing).any(-1) | (c.height != self.last.height)
        moving = torch.ones_like(self.valid)
        for mode in o.static_modes:
            moving &= c.mode != mode
        moving &= ((c.movement != self.last.movement).any(-1) | (c.speed != self.last.speed)
                   | (timed & (c.speed != 0)))
        return ~self.valid | changed | moving

    def on_device(self, commands):
        return JoystickBatch(**{f.name: getattr(commands, f.name).to(self.device) for f in fields(commands)})

    def plan(self, view, commands, ids, tick):
        o = self.owner
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        c = self.on_device(commands)
        valid = self.valid[ids]
        selected = JoystickBatch(**{f.name: torch.where(
            valid.reshape(-1, *([1]*(getattr(c, f.name).ndim-1))),
            getattr(c, f.name)[ids], getattr(self.initial, f.name)[ids]) for f in fields(c)})
        current = torch.cat((view.root_pos[ids].double(), view.root_rot[ids][:, [3, 0, 1, 2]].double(),
                             view.dof_pos[ids][:, self.to_planner].double()), -1)
        initial = current[:, None].expand(-1, o.context_frames, -1).clone()
        initial[..., :3] = self.root_xyz
        cursor = tick-self.start[ids]
        context = sample(self.q[ids], self.context_times[cursor.clamp(0, self.capacity)], o.hz, self.length[ids])
        context = torch.where(valid[:, None, None], context, initial)
        prediction, lengths = o.planner.plan_tensors(context, selected)
        new_length = self.resampled_lengths[lengths]
        times = torch_motion._time_grid(self.capacity, o.hz, self.device).expand(len(ids), -1)
        q = sample(prediction.double(), times, o.planner_hz, lengths)
        differences = torch.diff(q[..., 7:], dim=1)*o.hz
        vi = self.indices.expand(len(ids), -1).minimum(new_length[:, None]-2).clamp_min(0)
        v = gather(differences, vi)
        old_i = (cursor[:, None]+self.indices).clamp_min(0).minimum(self.length[ids, None]-1)
        new_i = (self.indices-o.lookahead).clamp_min(0).minimum(new_length[:, None]-1)
        old, new = gather(self.q[ids], old_i), gather(q, new_i)
        alpha = ((self.indices-o.lookahead).double()/o.blend_ticks).clamp(0, 1)[..., None]
        mixed = (1-alpha)*old + alpha*new
        mixed[..., 3:7] = torch_motion.slerp(old[..., 3:7], new[..., 3:7], alpha)
        mv = (1-alpha)*gather(self.v[ids], old_i) + alpha*gather(v, new_i)
        self.q[ids] = torch.where(valid[:, None, None], mixed, q)
        self.v[ids] = torch.where(valid[:, None, None], mv, v)
        self.length[ids] = new_length+valid.long()*o.lookahead
        self.start[ids] = tick
        self.polls[ids] = torch.where(valid, self.polls[ids], 0)
        self.valid[ids] = True
        self.last = JoystickBatch(**{f.name: getattr(self.last, f.name).clone() for f in fields(self.last)})
        for f in fields(self.last):
            getattr(self.last, f.name)[ids] = getattr(selected, f.name).to(getattr(self.last, f.name))
        o.replans += len(ids)
        return self.reference()

    def reference(self):
        torch._assert_async(self.valid.all(), 'Every agent needs a plan before publishing')
        q = torch.cat((self.q[..., :3], self.q[..., [4, 5, 6, 3]], self.q[..., 7:][..., self.to_package]), -1)
        return MotionReference(self.owner.robot_id, self.start.double()/self.owner.hz, float(self.owner.hz),
                               q, self.length.clone(), self.v[..., self.to_package].clone())
