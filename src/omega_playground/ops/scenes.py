"""Named single-agent deployment scenes; all paths are server-owned."""
from pathlib import Path


SCENES = {
    'campus': ('VibeCampus-v20261013', 'VibeCampus-v20261013/bridge-r10-v3', 'campus.json'),
    'obstacle': ('VibeObstacle-v20261103', 'vibeobstacle-v20261103', 'obstacle.json'),
}


def scene_id(path):
    name = Path(path).name.lower()
    return next((key for key in SCENES if name.startswith('vibe' + key)), None)


def resolve_scene(key, registry, package, avatar, armed):
    from omega_playground.map.navigation import CampusNavigation
    from omega_playground.scene.package import load
    if key not in SCENES:
        raise ValueError('Unknown scene')
    if registry is None:
        raise ValueError('切换场景需要离线资产注册表')
    scene_name, map_name, player_name = SCENES[key]
    scene_dir = Path('artifacts/scenes') / scene_name
    map_dir = Path('artifacts/maps') / map_name
    player = Path('configs/scenes') / player_name
    if not (map_dir / 'campus.png').is_file():
        raise ValueError(f'{key} 的二维地图尚未烘焙')
    scene = load(scene_dir)
    CampusNavigation(map_dir, scene.scene_sha256, scene.collision_sha256)
    assets = registry.select(package, scene.scene_sha256, avatar, armed, 1)
    if not player.is_file():
        raise ValueError('缺少场景出生点配置')
    return scene_dir, map_dir, player, assets
