from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from omega_playground.ops.initial_heading import initialize_route_heading


@pytest.mark.parametrize('offset', [0., -np.pi/2])
def test_route_frame_and_semantic_root_agree_for_multiple_teams(offset):
    world = SimpleNamespace(bridge=SimpleNamespace(heading_offset=offset), reset_agents=Mock())
    players = [SimpleNamespace(num_agents=3, reference=SimpleNamespace(
        sample=lambda t, i: (None, np.pi/2, None))),
        SimpleNamespace(num_agents=1, reference=SimpleNamespace(
            sample=lambda t, i: (None, 0., None)))]
    teams = SimpleNamespace(num_agents=4, players=players, ids=[np.arange(3), np.array([3])])
    facing = initialize_route_heading(world, teams, [{}, {'initial_facing_deg': 180}])
    ids = world.reset_agents.call_args.args[0]
    yaw = world.reset_agents.call_args.kwargs['yaw']
    assert ids == [0, 1, 2, 3]
    np.testing.assert_allclose(facing, [np.pi/2]*3+[np.pi])
    np.testing.assert_allclose(yaw+offset, facing)
