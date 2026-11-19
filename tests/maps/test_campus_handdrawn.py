import numpy as np
import torch
from types import SimpleNamespace

from omega_playground.control.drawn_path import drawn_path
from omega_playground.control.keyboard import KeyboardPlayer
from omega_playground.control.mission_player import MissionPlayer


def test_execution_keeps_preview_keypoints_and_reversal():
    keys = [[0, 0], [2, 1], [4, 0], [2, 1]]
    plan = drawn_path(keys)
    player = MissionPlayer({**plan, 'reference': {'preserve_path': True}}, 'cpu')
    assert len(player.reference.parts) == 2
    for key in keys:
        assert np.linalg.norm(player.reference.xy-key, axis=1).min() < 1e-8
    np.testing.assert_allclose(player.reference.parts[0].xy[-1], keys[-2])


def test_keyboard_uses_agent_heading_and_expires(monkeypatch):
    from omega_playground.control import keyboard
    player = KeyboardPlayer()
    now = [1.]
    monkeypatch.setattr(keyboard.time, 'monotonic', lambda: now[0])
    view = SimpleNamespace(root_rot=torch.tensor([[0., 0., .70710678, .70710678]]))
    player.update(1, 0)
    np.testing.assert_allclose(player.act(view).movement, [[0, 1, 0]], atol=1e-6)
    now[0] += 1.
    assert player.act(view).mode.item() == 0
