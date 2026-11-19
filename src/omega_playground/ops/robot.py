"""SONIC robot selection and workbench restart arguments."""
import sys


def restart_arguments(args, player, output):
    result = [sys.executable, '-m', 'omega_playground.cli', 'serve']
    for key in ('package', 'scene', 'map', 'device', 'port', 'render_hz',
                'render_width', 'render_height', 'camera', 'render_profile'):
        if getattr(args, key) is not None:
            result += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    for key in ('prepared_assets', 'physics_asset', 'visual_model', 'render_python', 'render_environment'):
        if getattr(args, key, None) is not None:
            result += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    if getattr(args, 'avatar', None):
        result += ['--avatar', args.avatar]
    result.append('--armed' if getattr(args, 'armed', False) else '--no-armed')
    result += ['--player', str(player), '--output', str(output)]
    for key in ('video', 'realtime'):
        if getattr(args, key, False):
            result.append('--' + key)
    if hasattr(args, 'video_bitrate_kbps'):
        result += ['--video-bitrate-kbps', str(args.video_bitrate_kbps)]
    if hasattr(args, 'video_codec'):
        result += ['--video-codec', args.video_codec]
    if getattr(args, 'dlss_frame_generation', 'off') != 'off':
        result += ['--dlss-frame-generation', args.dlss_frame_generation]
    return result


def policy_choice(package):
    from omega_playground.policy.registry import require_g1
    require_g1(package)
    return 'sonic'


def default_avatar(package):
    policy_choice(package)
    return 'g1'


def resolve_avatar(current, family, avatar):
    if family != policy_choice(current) or avatar != 'g1':
        raise ValueError('SONIC 支持 G1 外观')
    return current, None
