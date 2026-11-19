"""Exact WBC-boundary reference snapshots for offline causal replay, without pickle."""
import io
import json
import zipfile
from dataclasses import fields

import numpy as np
import torch

from omega_playground.policy.base import MotionReference


class ReferenceRecorder:
    def __init__(self, path, package):
        self.archive = zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED)
        self.meta = {'schema': 'omega.reference_archive/1', 'complete': False,
                     'policy_sha256': package.sha256sum(), 'robot_id': package.robot['id'],
                     'physics_hz': package.physics_hz, 'dof_names': package.dof_names,
                     'body_names': package.body_names, 'ticks': []}

    def append(self, tick):
        ref = tick.reference
        if ref is None:
            raise ValueError('Reference capture requires a motion-reference policy')
        if self.meta['ticks'] and tick.sim_time <= self.meta['ticks'][-1]['time']:
            raise ValueError('Reference tick times must increase strictly')
        values = {key: getattr(ref, key).detach().cpu().numpy().copy()
                  for key in ('qpos', 't0', 'length')}
        if ref.dof_vel is not None:
            values['dof_vel'] = ref.dof_vel.detach().cpu().numpy().copy()
        for obj, prefix in ((tick.command, 'command'), (tick.view, 'state')):
            for field in fields(obj):
                value = getattr(obj, field.name)
                values[f'{prefix}_{field.name}'] = (value.detach().cpu().numpy().copy()
                    if isinstance(value, torch.Tensor) else np.asarray(value))
        values['action'] = tick.action.detach().cpu().numpy().copy()
        values['pd_target'] = tick.pd_target.detach().cpu().numpy().copy()
        buffer = io.BytesIO()
        np.savez(buffer, **values)
        name = f'{len(self.meta["ticks"]):06d}.npz'
        self.archive.writestr(name, buffer.getvalue())
        self.meta['ticks'].append({'time': tick.sim_time, 'file': name,
                                   'fps': ref.fps, 'robot_id': ref.robot_id, 'schema': ref.schema})

    def finish(self):
        if not self.meta['ticks']:
            raise ValueError('Cannot finalize an empty reference recording')
        self.meta['complete'] = True

    def close(self):
        if self.archive.fp is not None:
            self.archive.writestr('index.json', json.dumps(self.meta))
            self.archive.close()


class ReferenceArchive:
    def __init__(self, path, package):
        self.archive = zipfile.ZipFile(path, 'r')
        try:
            self.meta = json.loads(self.archive.read('index.json'))
            if self.meta.get('schema') != 'omega.reference_archive/1' or not self.meta.get('complete'):
                raise ValueError('Incomplete or unsupported reference archive')
            if self.meta['policy_sha256'] != package.sha256sum():
                raise ValueError('Reference archive policy mismatch')
            if (self.meta['robot_id'] != package.robot['id'] or self.meta['dof_names'] != package.dof_names
                    or self.meta['body_names'] != package.body_names or self.meta['physics_hz'] != package.physics_hz):
                raise ValueError('Reference archive embodiment/cadence mismatch')
            self.times = np.array([tick['time'] for tick in self.meta['ticks']])
            if not len(self.times) or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
                raise ValueError('Invalid reference archive tick sequence')
        except BaseException:
            self.archive.close()
            raise

    def at(self, sim_time, device='cpu'):
        index = int(np.searchsorted(self.times, sim_time))
        if index and (index == len(self.times) or abs(self.times[index-1]-sim_time) < abs(self.times[index]-sim_time)):
            index -= 1
        if index == len(self.times) or abs(self.times[index]-sim_time) > 1e-7:
            raise ValueError('No recorded reference at this WBC tick')
        tick = self.meta['ticks'][index]
        with np.load(io.BytesIO(self.archive.read(tick['file'])), allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        def tensor(key):
            return torch.as_tensor(values[key], device=device)
        ref = MotionReference(tick['robot_id'], tensor('t0'), tick['fps'], tensor('qpos'), tensor('length'),
                              tensor('dof_vel') if 'dof_vel' in values else None, tick['schema'])
        return ref, values

    def close(self):
        self.archive.close()


class FrozenReferenceJoystick:
    """Bypass generator/retargeting; replay each exact recorded WBC condition."""
    def __init__(self, path, package, num_agents, device, initial_position_atol=1e-6):
        if not 0 < initial_position_atol <= 1e-5:
            raise ValueError('Initial position tolerance must be at most 10 micrometres')
        self.archive = ReferenceArchive(path, package)
        self.initial_position_atol = initial_position_atol
        self.schedule = package.layer_schedule('wbc')
        self.num_agents, self.device = num_agents, device
        self.checked_initial = False

    def should_replan(self, commands, sim_time):
        return torch.ones(self.num_agents, dtype=torch.bool, device=self.device)

    def plan(self, view, commands, agent_ids, sim_time):
        # Planner references are host tensors even for a CUDA tracker. The
        # tracker owns its upload; reference_window consumes NumPy views.
        reference, values = self.archive.at(sim_time, 'cpu')
        if not self.checked_initial:
            if abs(sim_time-self.archive.times[0]) > 1e-7:
                raise ValueError('Frozen replay must begin at its recorded initial tick')
            for field in fields(view):
                actual = getattr(view, field.name)
                expected = values['state_'+field.name]
                if isinstance(actual, torch.Tensor):
                    torch.testing.assert_close(actual, torch.as_tensor(expected, device=actual.device,
                                               dtype=actual.dtype),
                                               atol=self.initial_position_atol if field.name == 'body_pos' else 1e-6,
                                               rtol=0)
                elif actual != expected.item():
                    raise ValueError('Frozen replay initial-state metadata mismatch')
            if reference.qpos.shape[0] != self.num_agents:
                raise ValueError('Frozen replay agent count mismatch')
            self.checked_initial = True
        return reference

    def reset(self, agent_ids, view):
        raise ValueError('Frozen-reference causal replay does not support interactive reset')

    def close(self):
        self.archive.close()
