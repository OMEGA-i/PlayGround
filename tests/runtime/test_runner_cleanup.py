"""Simulation rebuild must close owned policy workers, including failed setup."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from omega_playground.policy.runner import Runner


def test_runner_closes_joystick_once():
    runner = object.__new__(Runner)
    worker = runner.joystick = Mock()
    runner.close()
    runner.close()
    worker.close.assert_called_once()


def test_runner_without_worker_can_close():
    runner = object.__new__(Runner)
    runner.joystick = SimpleNamespace()
    runner.close()
    runner.close()


def test_failed_runner_initialization_closes_worker(monkeypatch):
    from omega_playground.policy import registry
    worker = Mock()
    monkeypatch.setattr(registry, 'create_joystick', lambda *args: worker)
    monkeypatch.setattr(registry, 'create_wbc', lambda *args: SimpleNamespace(condition='joystick'))
    with pytest.raises(ValueError, match='does not match'):
        Runner(SimpleNamespace(physics_hz=120, num_agents=1),
               SimpleNamespace(physics_hz=120), None, torch.device('cpu'))
    worker.close.assert_called_once()
