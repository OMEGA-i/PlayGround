"""Align physical root orientation with the semantic route formation frame."""

import numpy as np


def initialize_route_heading(world, teams, specs):
    facing = np.empty(teams.num_agents)
    for player, ids, spec in zip(teams.players, teams.ids, specs, strict=True):
        override = spec.get('initial_facing_deg')
        facing[ids] = [player.reference.sample(0., i)[1] if override is None
                       else np.deg2rad(float(override)) for i in range(player.num_agents)]
    world.reset_agents(list(range(teams.num_agents)), yaw=facing-world.bridge.heading_offset)
    return facing
