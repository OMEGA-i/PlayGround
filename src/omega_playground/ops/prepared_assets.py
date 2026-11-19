"""Offline MJWarp bundle selection; never converts assets."""
import hashlib
import json
from pathlib import Path

def checked_metadata(root):
    root = Path(root).resolve()
    sums = json.loads((root/'SHA256SUMS.json').read_text())
    if 'asset.json' not in sums:
        raise ValueError('Prepared metadata is not checksummed')
    for name, digest in sums.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Prepared bundle path is missing or escapes its root')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Prepared bundle checksum mismatch: {name}')
    meta = json.loads((root/'asset.json').read_text())
    if meta.get('entry') not in sums:
        raise ValueError('Prepared entry is not checksummed')
    return meta


class PreparedAssetRegistry:
    def __init__(self, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if data.get('schema') != 'omega.prepared-runtime/1':
            raise ValueError('Unsupported prepared runtime registry')
        self.entries = {}
        for entry in data['entries']:
            if type(entry['armed']) is not bool or type(entry['num_agents']) is not int or not 1 <= entry['num_agents'] <= 9:
                raise ValueError('Invalid prepared population selector')
            key = (entry['policy_sha256'], entry['scene_sha256'], entry['avatar'], entry['armed'], entry['num_agents'])
            if key in self.entries:
                raise ValueError('Duplicate prepared runtime selector')
            self.entries[key] = {name: (path.parent/entry[name]).resolve()
                                 for name in ('physics_asset', 'visual_model')}
            if entry.get('render_environment'):
                self.entries[key]['render_environment'] = (path.parent/entry['render_environment']).resolve()
            if 'runtime_profile' in entry:
                raise ValueError('Unsupported legacy runtime profile')

    def select(self, package, scene_sha256, avatar, armed, count):
        key = (package.sha256sum(), scene_sha256, avatar, armed, count)
        if key not in self.entries:
            raise ValueError('该策略、外观、Armed 和人数组合尚未准备离线资产')
        paths = self.entries[key]
        physics = checked_metadata(paths['physics_asset'])
        visual = checked_metadata(paths['visual_model'])
        if physics.get('policy_sha256') != key[0] or not physics.get('joint_physics'):
            raise ValueError('Prepared physics policy contract mismatch')
        if (visual.get('policy_sha256') != key[0] or visual.get('scene_sha256') != scene_sha256
                or visual.get('num_agents', 1) != count or bool(visual.get('armed')) != armed):
            raise ValueError('Prepared visual selection mismatch')
        from omega_playground.ops.robot import default_avatar
        if visual.get('schema') != 'omega.mjwarp-visual/1' or avatar != default_avatar(package):
            raise ValueError('Prepared visual avatar/schema mismatch')
        if 'render_environment' in paths:
            environment = checked_metadata(paths['render_environment'])
            if (environment.get('schema') != 'omega.mjwarp-environment/1'
                    or environment.get('scene_sha256') != scene_sha256):
                raise ValueError('Prepared environment scene/schema mismatch')
        result = {name: str(value) for name, value in paths.items()}
        result.setdefault('render_environment', None)
        return result
