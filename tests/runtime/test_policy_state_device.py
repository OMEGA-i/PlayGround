from dataclasses import fields

import pytest
import torch

from omega_playground.policy.base import RobotBatchView


def view():
    return RobotBatchView(torch.randn(2, 3, 3), torch.randn(2, 3, 4),
                          torch.randn(2, 3, 3), torch.randn(2, 3, 3),
                          torch.randn(2, 5), torch.randn(2, 5), torch.zeros(2), 1, .7)


def test_same_device_preserves_state_and_storage():
    state = view()
    assert state.to('cpu') is state


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA transfer requires GPU')
def test_cpu_physics_state_can_feed_gpu_policy_and_return_without_reordering():
    state = view()
    moved = state.to('cuda:0')
    restored = moved.to('cpu')
    assert moved.anchor == state.anchor and moved.heading_offset == state.heading_offset
    for field in fields(state):
        original = getattr(state, field.name)
        if isinstance(original, torch.Tensor):
            assert getattr(moved, field.name).device.type == 'cuda'
            assert original.device.type == 'cpu'
            torch.testing.assert_close(getattr(restored, field.name), original, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Mixed-device Runner requires GPU')
def test_runner_keeps_host_state_and_handles_off_player_tick_reset(monkeypatch):
    from types import SimpleNamespace as NS

    from omega_playground.policy import registry
    from omega_playground.policy.base import JoystickBatch
    from omega_playground.policy.runner import Runner
    from omega_playground.policy.schedule import Schedule

    state = view()
    state.body_rot.zero_()
    state.body_rot[..., 3] = 1
    device = torch.device('cuda:0')
    command = JoystickBatch(torch.ones(2, device=device, dtype=torch.long),
                            torch.ones(2, device=device), torch.ones(2, 3, device=device),
                            torch.ones(2, 3, device=device), torch.zeros(2, device=device),
                            torch.zeros(2, device=device))
    seen = []
    def act(actual, condition, time):
        assert actual is state and actual.dof_pos.device.type == 'cpu'
        seen.append(condition.speed.clone())
        return torch.zeros(2, 5, device=device)
    schedule = lambda period: Schedule.from_json({'trigger': 'periodic', 'period_s': period})
    monkeypatch.setattr(registry, 'create_joystick', lambda *args: None)
    monkeypatch.setattr(registry, 'create_wbc', lambda *args: NS(
        condition='joystick', schedule=schedule(.1), act=act))
    world = NS(physics_hz=10, num_agents=2, physics_steps=0, sim_time=0., robot_view=lambda: state,
               set_pd_targets=lambda target: None)
    def step():
        world.physics_steps += 1
        world.sim_time = world.physics_steps/10
    world.step = step
    package = NS(physics_hz=10, manifest={'action': {'transform': 'clamp', 'offset': [0.]*5, 'scale': [1.]*5}})
    player = NS(schedule=schedule(.2), act=lambda *args: command)
    runner = Runner(world, package, player, device)
    runner.step()
    runner.request_reset(torch.tensor([1], device=device))
    runner.step()
    torch.testing.assert_close(seen[1], torch.tensor([1., -1.], device=device))
    assert not runner.pending_reset.any()
