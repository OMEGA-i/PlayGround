from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from omega_playground.control.mission_player import MissionPlayer
from omega_playground.control.route_player import RoutePlayer
from omega_playground.ops.runtime import OperationsRuntime


def make_player(mission):
    spec = {'route': [[0, 0], [10, 0]], 'schedule': {'trigger': 'periodic', 'period_s': .1}}
    return MissionPlayer(spec, 'cpu') if mission else RoutePlayer(spec, 1, 'cpu')


def check(player, position, t):
    player.check(SimpleNamespace(root_pos=torch.tensor([[*position, 1.]])), t)


@pytest.mark.parametrize('mission', [False, True])
def test_offline_divergence_still_holds(mission):
    player = make_player(mission)
    check(player, [20., 0.], 0.)
    check(player, [20., 0.], 2.)
    assert player.hold['code'] == 'tracking_divergence'


@pytest.mark.parametrize('mission', [False, True])
def test_web_warns_once_and_recovers_without_reset(tmp_path, mission):
    player = make_player(mission)
    server = SimpleNamespace(event=Mock())
    runtime = OperationsRuntime(SimpleNamespace(sim_time=2., physics_steps=400), None,
                                SimpleNamespace(players=[player], ids=[np.array([7])]), server, tmp_path)
    try:
        runtime.paused = False
        check(player, [20., 0.], 0.)
        check(player, [20., 0.], 2.)
        runtime.publish_warnings()
        check(player, [21., 0.], 3.)
        runtime.publish_warnings()
        assert player.hold is None and runtime.hold is None and not runtime.paused
        assert server.event.call_count == 1
        assert server.event.call_args.args[0]['kind'] == 'warning'
        assert server.event.call_args.args[0]['agents'] == [7]
        check(player, player.targets(4.)[0][0], 4.)
        runtime.publish_warnings()
        assert server.event.call_count == 2
        assert server.event.call_args.args[0]['kind'] == 'warning_cleared'
        assert player.warning is None
        player.navigation = SimpleNamespace(valid=lambda *args: np.array([False]))
        check(player, [20., 0.], 5.)
        assert player.hold is None
        assert any(w['code'] == 'off_navigable' for w in player.warnings)
    finally:
        runtime.close()
