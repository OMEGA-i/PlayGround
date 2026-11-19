"""Shared rule Players for the supported SONIC/SOMA pipelines."""
from copy import deepcopy
import os


def device_player_enabled(device):
    override = os.environ.get('OMEGA_GPU_PLAYER')
    if override not in (None, '0', '1'):
        raise ValueError('OMEGA_GPU_PLAYER must be 0 or 1')
    return override == '1' if override is not None else str(device).split(':')[0] == 'cuda'


def configured(player, spec):
    # Preserve tuning across interactive task replacement, including the first
    # replacement of a startup RoutePlayer by a formation MissionPlayer.
    player.formation_settings = deepcopy(spec.get("formation", {}))
    return player


def mission_player(family, spec, device, count=1):
    if family != 'sonic' or count != 1:
        raise ValueError('The native workflow requires one SONIC robot')
    from omega_playground.control.mission_player import MissionPlayer
    if device_player_enabled(device):
        from omega_playground.control.tensor_web_player import TensorWebPlayer
        return configured(TensorWebPlayer(spec, device), spec)
    return configured(MissionPlayer(spec, device), spec)


def route_player(family, spec, count, device):
    if family != 'sonic' or count != 1:
        raise ValueError('The native workflow requires one SONIC robot')
    from omega_playground.control.route_player import RoutePlayer
    if device_player_enabled(device):
        from omega_playground.control.tensor_route_player import TensorRoutePlayer
        return configured(TensorRoutePlayer(spec, device), spec)
    return configured(RoutePlayer(spec, 1, device), spec)
