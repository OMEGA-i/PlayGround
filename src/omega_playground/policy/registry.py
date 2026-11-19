"""Policy family -> adapter. Adding a family adds one entry here and nothing in the runner."""

import torch

from omega_playground.policy.package import PackageError, PolicyPackage


def require_g1(package):
    if package.family != 'sonic' or not package.robot['id'].startswith('unitree_g1'):
        raise PackageError('The SONIC workflow requires a G1 policy package')


def create_wbc(package: PolicyPackage, num_agents: int, device: torch.device):
    from omega_playground.policy.sonic.tracker import SonicTracker
    require_g1(package)
    return SonicTracker(package, num_agents, device)


def create_joystick(package: PolicyPackage, num_agents: int, device: torch.device):
    from omega_playground.policy.sonic.planner import SonicJoystick
    require_g1(package)
    return SonicJoystick(package, num_agents, device)
