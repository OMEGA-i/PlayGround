import os
from copy import copy
from dataclasses import fields
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from omega_playground.policy.base import JoystickBatch
from omega_playground.policy.package import load
from omega_playground.policy.sonic.planner import SonicJoystick


def command(n, device):
    return JoystickBatch(torch.full((n,), 2, device=device), torch.full((n,), .65, device=device),
        torch.tensor([[1., 0., 0.]], device=device).expand(n, -1).clone(),
        torch.tensor([[1., 0., 0.]], device=device).expand(n, -1).clone(),
        torch.zeros(n, device=device), torch.full((n,), -1., device=device))


def test_real_device_joystick_partial_replans_no_host_readback(monkeypatch):
    path = os.environ.get('OMEGA_TEST_SONIC_PACKAGE')
    if not path or not torch.cuda.is_available():
        pytest.skip('CUDA and OMEGA_TEST_SONIC_PACKAGE required')
    package = load(path)
    n, device = 9, torch.device('cuda:0')
    actual = SonicJoystick(package, n, device)
    oracle = copy(actual)
    oracle.device_buffers = None
    view = NS(root_pos=torch.tensor([[0., 0., .8]], device=device).expand(n, -1),
              root_rot=torch.tensor([[0., 0., 0., 1.]], device=device).expand(n, -1),
              dof_pos=torch.tensor(package.robot['default_dof_pos'], device=device).expand(n, -1))
    cmds = command(n, device)

    def forbidden(*args, **kwargs):
        raise AssertionError('Device Joystick must not read back to host')

    for step in range(14):
        t = step*.1
        if step == 3:
            cmds.speed[0] = .3
        if step == 5:
            cmds.facing[1] = cmds.facing.new_tensor([0., 1., 0.])
        if step == 7:
            ids = torch.tensor([2, 5], device=device)
            actual.reset(ids, view)
            oracle.reset(ids, view)
        if step == 9:
            cmds.mode[3] = 0
            cmds.speed[3] = -1
        mask_expected = oracle.should_replan(cmds, t)
        with monkeypatch.context() as patch:
            for name in ('cpu', 'numpy', 'item', 'tolist'):
                patch.setattr(torch.Tensor, name, forbidden)
            mask = actual.should_replan(cmds, t)
        torch.testing.assert_close(mask.cpu(), mask_expected)
        ids = mask.nonzero().flatten()
        if not len(ids):
            continue
        expected = oracle.plan(view, cmds, ids, t)
        with monkeypatch.context() as patch:
            for name in ('cpu', 'numpy', 'item', 'tolist'):
                patch.setattr(torch.Tensor, name, forbidden)
            result = actual.plan(view, cmds, ids, t)
        for f in fields(result):
            value = getattr(result, f.name)
            if isinstance(value, torch.Tensor):
                assert value.is_cuda
        torch.testing.assert_close(result.length, expected.length)
        torch.testing.assert_close(result.t0, expected.t0)
        for i, length in enumerate(expected.length.cpu().tolist()):
            np.testing.assert_allclose(result.qpos[i, :length].cpu(), expected.qpos[i, :length].cpu(), atol=2e-6)
            np.testing.assert_allclose(result.dof_vel[i, :length].cpu(), expected.dof_vel[i, :length].cpu(), atol=2e-5)
