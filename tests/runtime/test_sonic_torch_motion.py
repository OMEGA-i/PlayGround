"""Compare device motion arithmetic to the release SciPy oracle (no physics)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omega_playground.policy.sonic import motion, torch_motion

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))]


def clip(count, seed=4):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(count, 36))
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=-1, keepdims=True)
    # Identical, antipodal and almost identical rotations exercise Slerp limits.
    q[1, 3:7] = q[0, 3:7]
    q[2, 3:7] = -q[0, 3:7]
    q[3, 3:7] = q[2, 3:7] + 1e-12
    return q


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('count', [24, 40, 44])
@pytest.mark.parametrize('batch', [1, 3, 6, 9])
def test_resample_matches_release(device, count, batch):
    source = np.stack([clip(count, i) for i in range(batch)])
    q, v = torch_motion.resample(torch.tensor(source, device=device), 30, 50)
    expected = [motion.resample(row, 30, 50) for row in source]
    np.testing.assert_allclose(q.cpu(), np.stack([r[0] for r in expected]), atol=2e-12)
    np.testing.assert_allclose(v.cpu(), np.stack([r[1] for r in expected]), atol=2e-11)
    assert q.device.type == v.device.type == device
    torch.testing.assert_close(v[:, -1], v[:, -2])


@pytest.mark.parametrize('device', DEVICES)
def test_context_and_blend_match_release(device):
    old, ov = motion.resample(clip(40), 30, 50)
    new, nv = motion.resample(clip(44, 7), 30, 50)
    times = np.r_[-1., np.arange(4)/30, np.arange(len(old))/50, 100.]
    sampled = torch_motion.sample_qpos(torch.tensor(old, device=device), times, 50)
    np.testing.assert_allclose(sampled.cpu(), motion.sample_qpos(old, times, 50), atol=2e-12)
    for cursor in (0, 17, 100):
        q, v = torch_motion.blend(*(torch.tensor(x, device=device) for x in (old, ov)), cursor,
                                 *(torch.tensor(x, device=device) for x in (new, nv)), 2, 8)
        eq, ev = motion.blend(old, ov, cursor, new, nv, 2, 8)
        np.testing.assert_allclose(q.cpu(), eq, atol=2e-12)
        np.testing.assert_allclose(v.cpu(), ev, atol=2e-11)


@pytest.mark.parametrize('device', DEVICES)
def test_motion_math_never_reads_back_to_host(device, monkeypatch):
    q = torch.tensor(np.stack([clip(40)]*9), device=device)

    def forbidden(*args, **kwargs):
        raise AssertionError('Motion interpolation read back to host')

    with monkeypatch.context() as patch:
        for name in ('cpu', 'numpy', 'item', 'tolist'):
            patch.setattr(torch.Tensor, name, forbidden)
        result, velocity = torch_motion.resample(q, 30, 50)
        torch_motion.blend(result[0], velocity[0], 5, result[1], velocity[1], 2, 8)
    assert result.shape == (9, 66, 36)


@pytest.mark.parametrize('device', DEVICES)
def test_joystick_device_buffers_match_host_after_partial_replan_and_reset(device, monkeypatch):
    from pathlib import Path

    from omega_playground.policy.base import JoystickBatch
    from omega_playground.policy.package import PolicyPackage
    from omega_playground.policy.sonic.planner import SonicJoystick

    class Planner:
        max_frames = 44

        def __init__(self, package, *args):
            self.gpu = package.manifest['gpu']

        def plan_tensors(self, contexts, commands):
            lengths = torch.where(commands.speed <= 0, 40, 44)
            return torch.tensor(clip(44), device=device).expand(len(contexts), -1, -1), lengths

        def plan_batch(self, contexts, commands, *, return_tensors=False):
            # Different clip lengths, quaternion signs, all independent rows.
            rows = [clip(40 if c.speed <= 0 else 44) for c in commands]
            return [torch.tensor(q, device=device) for q in rows] if return_tensors else rows

    monkeypatch.setattr('omega_playground.policy.sonic.planner.SonicPlanner', Planner)
    joints = [f'joint{i}' for i in range(29)]
    manifest = {'robot': {'id': 'g1', 'dof_names': joints[::-1]}, 'modules': {'planner': {
        'qpos_layout': {'joint_names': joints}}}, 'layers': {'joystick': {
        'schedule': {'trigger': 'periodic', 'period_s': .1}, 'output': {'fps': 50},
        'release_motion': {'planner_output_hz': 30, 'lookahead_ticks': 2, 'blend_ticks': 8,
            'context_frames': 4, 'static_modes': [0], 'replan_poll_count': {'default': 10},
            'initial_command': {'mode': 0, 'speed': 0., 'height': -1.,
                                'movement': [1, 0, 0], 'facing': [1, 0, 0]},
            'initial_context_root_xyz': [0, 0, .8]}}}}
    host = SonicJoystick(PolicyPackage(Path('.'), {**manifest, 'gpu': False}), 3, 'cpu')
    gpu = SonicJoystick(PolicyPackage(Path('.'), {**manifest, 'gpu': True}), 3, device)
    view = SimpleNamespace(root_pos=torch.zeros(3, 3, device=device),
        root_rot=torch.tensor([[0., 0., 0., 1.]]*3, device=device), dof_pos=torch.zeros(3, 29, device=device))
    command = JoystickBatch(torch.zeros(3), torch.tensor([1., 0., 1.]), torch.zeros(3, 3),
                            torch.tensor([[1., 0., 0.]]*3), torch.zeros(3), torch.full((3,), -1.))
    for step, ids in enumerate(([0, 1, 2], [0, 2], [1])):
        if step == 2:
            gpu.reset(torch.tensor(ids), view)
            host.reset(torch.tensor(ids), view)
        actual = gpu.plan(view, command, torch.tensor(ids), step*.1)
        expected = host.plan(view, command, torch.tensor(ids), step*.1)
        assert actual.qpos.device.type == device
        for i, length in enumerate(expected.length.tolist()):
            np.testing.assert_allclose(actual.qpos[i, :length].cpu(), expected.qpos[i, :length], atol=2e-12)
            np.testing.assert_allclose(actual.dof_vel[i, :length].cpu(), expected.dof_vel[i, :length], atol=2e-11)
        torch.testing.assert_close(actual.length.cpu(), expected.length)
        torch.testing.assert_close(actual.t0.cpu(), expected.t0)


def test_real_cuda_planner_device_output_matches_host_api():
    import os

    from omega_playground.policy.package import load
    from omega_playground.policy.sonic.planner import Command, SonicPlanner

    path = os.environ.get('OMEGA_TEST_SONIC_PACKAGE')
    if not path or not torch.cuda.is_available():
        pytest.skip('Set OMEGA_TEST_SONIC_PACKAGE on a CUDA host for release-weight verification')
    package = load(path)
    planner = SonicPlanner(package, 9, 'cuda:0')
    rm = package.manifest['layers']['joystick']['release_motion']
    context = np.zeros((4, 36), np.float32)
    context[:, :3], context[:, 3] = rm['initial_context_root_xyz'], 1.
    joints = package.manifest['modules']['planner']['qpos_layout']['joint_names']
    context[:, 7:] = np.asarray(package.robot['default_dof_pos'])[[package.dof_names.index(j) for j in joints]]
    for count in (1, 3, 6, 9):
        commands = [Command(0 if i%2 else 2, 0. if i%2 else .65, -1., (1., 0., 0.), (1., 0., 0.))
                    for i in range(count)]
        host = planner.plan_batch([context]*count, commands)
        device = planner.plan_batch([torch.tensor(context, device='cuda')]*count, commands, return_tensors=True)
        for expected, actual in zip(host, device, strict=True):
            assert actual.is_cuda
            np.testing.assert_array_equal(actual.cpu(), expected)
            q, v = torch_motion.resample(actual, 30, 50)
            eq, ev = motion.resample(expected, 30, 50)
            np.testing.assert_allclose(q.cpu(), eq, atol=2e-7)
            np.testing.assert_allclose(v.cpu(), ev, atol=2e-7)
        saved = [q.clone() for q in device]
        planner.plan_batch([context]*count, commands)
        for q, before in zip(device, saved, strict=True):
            torch.testing.assert_close(q, before, rtol=0, atol=0)
