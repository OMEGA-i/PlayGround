"""Batched native Tracker execution: independent MuJoCo worlds, one actor forward."""

import numpy as np
import torch

from .chunk import streaming_window
from .rollout import TrackingSession


class BatchedTrackingSession(TrackingSession):
    def __init__(self, robot, checkpoint, xml, num_envs):
        if num_envs < 1:
            raise ValueError('num_envs must be positive')
        super().__init__(robot, checkpoint, xml, num_envs=num_envs)
        self.num_envs = num_envs
        self.queues = [None] * num_envs
        self.initial_states = [None] * num_envs

    @torch.inference_mode()
    def append_one(self, index, reference, *, reset=False):
        joint_names = list(reference['joint_names'])
        if (str(reference['embodiment']) != self.robot or int(reference['fps']) != 50
                or len(set(joint_names)) != len(joint_names) or set(joint_names) != set(self.names[0])):
            raise ValueError('Batched reference schema mismatch')
        order = [joint_names.index(name) for name in self.names[0]]
        q = reference['qpos']
        root = np.c_[q[:, :7], reference['root_linear_velocity'], reference['root_angular_velocity_world']]
        tensors = [torch.as_tensor(x, device=self.device, dtype=torch.float32) for x in
                   (root, q[:, 7:][:, order], reference['joint_velocity'][:, order])]
        clip = torch.cat([*tensors, self.fk(*tensors).flatten(1)], dim=1)
        self.queues[index] = clip
        if reset:
            self.initial_states[index] = clip[0].clone()
            self.env.reset(torch.tensor([index], device=self.device), clip[:1])
            previous = clip.new_zeros((1, len(self.names[0])))
            obs = self.env.layout.obs(self.env.state()[index:index+1], previous, self.env.default,
                                      False, goal=clip[:1])
            if self.history is None:
                self.history = obs.new_zeros((self.num_envs, 10, obs.shape[-1]))
            self.history[index] = obs[:, None].repeat(1, 10, 1)[0]
        return clip

    @torch.inference_mode()
    def append_batch(self, references, *, reset=False):
        if len(references) != self.num_envs:
            raise ValueError('One independent reference is required for each world')
        clips = []
        for reference in references:
            joint_names = list(reference['joint_names'])
            if (str(reference['embodiment']) != self.robot or int(reference['fps']) != 50
                    or len(set(joint_names)) != len(joint_names)
                    or set(joint_names) != set(self.names[0])):
                raise ValueError('Batched reference schema mismatch')
            order = [joint_names.index(name) for name in self.names[0]]
            q = reference['qpos']
            root = np.c_[q[:, :7], reference['root_linear_velocity'], reference['root_angular_velocity_world']]
            tensors = [torch.as_tensor(x, device=self.device, dtype=torch.float32) for x in
                       (root, q[:, 7:][:, order], reference['joint_velocity'][:, order])]
            clips.append(torch.cat([*tensors, self.fk(*tensors).flatten(1)], dim=1))
        self.queues = clips
        if reset:
            self.initial_states = [clip[0].clone() for clip in clips]
            first = torch.stack(self.initial_states)
            self.env.reset(torch.arange(self.num_envs, device=self.device), first)
            previous = first.new_zeros((self.num_envs, len(self.names[0])))
            obs = self.env.layout.obs(self.env.state(), previous, self.env.default, False, goal=first)
            self.history = obs[:, None].repeat(1, 10, 1)
        elif self.history is None:
            raise ValueError('First batch must reset all worlds')
        return clips

    @torch.inference_mode()
    def step_batch(self, *, measured_heading=True, active=None):
        from supermimic.core import yawq, qconj, qmul
        from supermimic.simulation_health import NonfiniteSimulationState

        inactive = []
        if active is not None:
            inactive = [i for i in range(self.num_envs) if i not in active]
            for i in inactive:
                hold = self.initial_states[i][None].clone()
                hold[:, 7:13] = 0
                hold[:, 13+len(self.names[0]):13+2*len(self.names[0])] = 0
                self.queues[i] = hold.repeat(2, 1)
            if inactive:
                ids = torch.tensor(inactive, device=self.device)
                self.env.reset(ids, torch.stack([self.queues[i][0] for i in inactive]))
        windows, masks = zip(*(streaming_window(q, len(self.names[0])) for q in self.queues))
        window = torch.stack(windows)
        self.lookahead_valid = torch.stack(masks)
        if measured_heading:
            actual = self.env.view('xquat')[:, self.env.root_bid]
            target = window[:, 0, 3:7]
            window[:, 0, 3:7] = qmul(yawq(actual), qmul(qconj(yawq(target)), target))
        action = self.actor(self.history, window)
        previous = self.env.state().clone()
        invalid_action = ~torch.isfinite(action).all(-1)
        action[invalid_action] = 0
        if inactive:
            action[inactive] = 0
        try:
            state = self.env.step(action)
        except NonfiniteSimulationState as error:
            state = error.state
        self.invalid = invalid_action | ~torch.isfinite(state).all(-1)
        # Placeholders are only for internal batch arithmetic. Invalid transitions
        # are thrown into their recorder and never appended to published data.
        safe_state = torch.where(self.invalid[:, None], previous, state)
        self.last_action = action.detach().clone()
        self.last_pd_target = self.env.view('ctrl').detach().clone()
        self.queues = [q[1:] for q in self.queues]
        target = torch.stack([q[0] for q in self.queues])
        failed = self.env.layout.done(safe_state, target) | self.invalid
        obs = self.env.layout.obs(safe_state, action, self.env.default, False, goal=target)
        self.history = torch.cat((self.history[:, 1:], obs[:, None]), 1)
        return state, failed

    def physical_poses(self):
        xyz = self.env.view('xpos')[:, 1:]
        wxyz = self.env.view('xquat')[:, 1:]
        return torch.cat((xyz, wxyz[:, :, [1, 2, 3, 0]]), -1)
