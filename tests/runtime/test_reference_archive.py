from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omega_playground.eval.reference_archive import (
    FrozenReferenceJoystick,
    ReferenceArchive,
    ReferenceRecorder,
)
from omega_playground.policy.base import JoystickBatch, MotionReference
from omega_playground.policy.runner import WbcTick
from omega_playground.policy.base import RobotBatchView


def package(digest='same'):
    return SimpleNamespace(sha256sum=lambda: digest, robot={'id': 'test'}, physics_hz=1000,
                           dof_names=['joint'], body_names=['root'])


@pytest.mark.parametrize('with_rates', [True, False])
def test_exact_reference_snapshot_preserves_window_dtype_and_initial_state(tmp_path, with_rates):
    path = tmp_path/'reference.zip'
    q = torch.arange(32, dtype=torch.float64).reshape(1, 4, 8)
    ref = MotionReference('test', torch.tensor([-.1], dtype=torch.float64), 50., q,
                          torch.tensor([3]), torch.ones(1, 4, 1) if with_rates else None)
    command = JoystickBatch(torch.zeros(1, dtype=torch.int64), torch.tensor([-1.]),
        torch.zeros(1, 3), torch.tensor([[1., 0., 0.]]), torch.zeros(1), torch.tensor([-1.]))
    pos = torch.tensor([[[3., 4., 1.]]])
    rot = torch.tensor([[[0., 0., 0., 1.]]])
    state = RobotBatchView(pos, rot, torch.zeros_like(pos), torch.zeros_like(pos),
                          torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1))
    writer = ReferenceRecorder(path, package())
    try:
        writer.append(WbcTick(0., state, command, ref, torch.ones(1, 1), torch.zeros(1, 1)))
        q.add_(100)  # Mutation after capture must not alter the frozen first window.
        writer.append(WbcTick(.02, state, command, ref, torch.ones(1, 1), torch.zeros(1, 1)))
        writer.finish()
    finally:
        writer.close()
    archive = ReferenceArchive(path, package())
    try:
        original, values = archive.at(0.)
        torch.testing.assert_close(original.qpos, q-100)
        torch.testing.assert_close(original.t0, ref.t0)
        assert original.length.tolist() == [3]
        assert (original.dof_vel is not None) == with_rates
        np.testing.assert_array_equal(values['state_body_pos'], state.body_pos.numpy())
        np.testing.assert_array_equal(values['command_speed'], [-1.])
        torch.testing.assert_close(archive.at(.02+1e-10)[0].qpos, q)
        with pytest.raises(ValueError, match='WBC tick'):
            archive.at(.01)
        with pytest.raises(ValueError, match='WBC tick'):
            archive.at(1.)
    finally:
        archive.close()
    with pytest.raises(ValueError, match='policy mismatch'):
        ReferenceArchive(path, package('different'))


def test_interrupted_recording_is_not_accepted_as_complete(tmp_path):
    path = tmp_path/'reference.zip'
    writer = ReferenceRecorder(path, package())
    writer.close()
    with pytest.raises(ValueError, match='Incomplete'):
        ReferenceArchive(path, package())


def test_frozen_replay_checks_initial_state_then_ignores_live_commands(tmp_path):
    path = tmp_path/'reference.zip'
    pkg = package()
    pkg.layer_schedule = lambda layer: ('recorded', layer)
    pos = torch.tensor([[[3., 4., 1.]]])
    rot = torch.tensor([[[0., 0., 0., 1.]]])
    state = RobotBatchView(pos, rot, torch.zeros_like(pos), torch.zeros_like(pos),
                          torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1))
    command = JoystickBatch(torch.zeros(1, dtype=torch.int64), torch.tensor([-1.]),
        torch.zeros(1, 3), torch.tensor([[1., 0., 0.]]), torch.zeros(1), torch.tensor([-1.]))
    ref = MotionReference('test', torch.zeros(1), 50., torch.zeros(1, 4, 8), torch.tensor([4]))
    writer = ReferenceRecorder(path, pkg)
    try:
        for time in (0., .02):
            writer.append(WbcTick(time, state, command, ref, torch.ones(1, 1), torch.zeros(1, 1)))
        writer.finish()
    finally:
        writer.close()
    joystick = FrozenReferenceJoystick(path, pkg, 1, 'cpu')
    try:
        from dataclasses import replace
        wrong_state = replace(state, body_pos=state.body_pos + 1.)
        with pytest.raises(AssertionError):
            joystick.plan(wrong_state, command, [0], 0.)
        with pytest.raises(ValueError, match='initial tick'):
            joystick.plan(state, command, [0], .02)
        torch.testing.assert_close(joystick.plan(state, command, [0], 0.).qpos, ref.qpos)
        command.speed.fill_(10)
        # Tracker compute device must not move NumPy-consumed reference buffers.
        joystick.device = 'cuda:0'
        torch.testing.assert_close(joystick.plan(wrong_state, command, [0], .02).qpos, ref.qpos)
        joystick.device = 'cpu'
        assert joystick.should_replan(command, .02).tolist() == [True]
        with pytest.raises(ValueError, match='interactive reset'):
            joystick.reset([0], state)
    finally:
        joystick.close()
