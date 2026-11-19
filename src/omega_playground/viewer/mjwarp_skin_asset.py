"""Checksummed offline skin topology; loading never invokes a model compiler."""
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from .mjwarp_skin import SkinMeshScene
from .skin import SkinRenderer
from .skin import load as load_skin


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Skin asset path missing or outside bundle')
    return path


def prepare_skin(package, skin_path, output, *, agents=1, armed=False, scene_path=None, flip_texture_v=None):
    if type(agents) is not int or not 1 <= agents <= 9:
        raise ValueError('Skin agent count must be between one and nine')
    skin = load_skin(skin_path)
    if flip_texture_v is None:
        # These authored Blender exports retain bottom-origin UVs, while the
        # compiled PNG texels are stored top-row first. Keep other formats explicit.
        flip_texture_v = skin.meta.get('appearance') in (
            'ue5_manny', 'ue4_mannequin', 'UE5 Manny', 'UE4 Mannequin')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    copied = output/'skin'
    copied.mkdir()
    sums = json.loads((skin.root/'SHA256SUMS.json').read_text())
    files = [*sums, 'SHA256SUMS.json']
    # Grip preparation intentionally leaves the base skin manifest untouched.
    # Include these derived arrays in the outer bundle checksum, and let the
    # normal grip loader verify their embedded source digest before use.
    if armed:
        files.extend(name for name in ('armed_skin.npz', 'armed_robot_skin.npz')
                     if (skin.root/name).is_file() and name not in sums)
    for name in files:
        source = checked_path(skin.root, name)
        target = copied/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Only FK for the compilation pose, never a physics step.
    physical = mujoco.MjModel.from_xml_path(str(package.asset_path))
    data = mujoco.MjData(physical)
    mujoco.mj_kinematics(physical, data)
    ids = [physical.body(name).id for name in package.body_names]
    one = np.c_[data.xpos[ids], data.xquat[ids][:, [1, 2, 3, 0]]]
    poses = np.concatenate([one.copy() for _ in range(agents)])
    np.save(output/'initial_poses.npy', poses)
    painter = SkinRenderer(load_skin(copied), package.body_names, package.asset_path, agents, 'cpu', armed=armed)
    if armed:
        shutil.copyfile(painter.equipment_path, output/'equipment.npz')
    bridge = SkinMeshScene(flip_texture_v=flip_texture_v,
                           constant_linear_colors=skin.meta.get('source_type') == 'rigid_robot_shell')
    painter.log(bridge, poses)
    spec = mujoco.MjSpec()
    scene_meta = {'scene_sha256': None, 'instance_transforms': [], 'visual_gaps': []}
    spec.worldbody.add_light(pos=[0, -3, 5], dir=[0, .5, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    if scene_path is None:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .1], rgba=[.22, .24, .27, 1])
    else:
        from omega_playground.scene.package import load as load_scene
        from omega_playground.viewer.usd_scene import add_usd_scene
        scene = load_scene(scene_path)
        scene_meta.update(add_usd_scene(spec, scene.entry, output/'textures'))
        scene_meta['scene_sha256'] = scene.scene_sha256
        from omega_playground.viewer.usd_lighting import add_usd_lighting
        scene_meta['lighting'] = add_usd_lighting(spec, scene.entry)
        scene_meta['visual_gaps'].extend(scene_meta['lighting']['visual_gaps'])
    spec.worldbody.add_camera(name='qc', fovy=np.degrees(2*np.arctan(1.25/4)))
    model = bridge.compile(spec)
    mujoco.mj_saveModel(model, str(output/'visual.mjb'))
    meta = {**scene_meta, 'schema': 'omega.mjwarp-skin/1', 'mujoco_version': mujoco.__version__,
            'policy_sha256': package.sha256sum(), 'policy_body_names': package.body_names,
            'skin_manifest_sha256': digest(copied/'SHA256SUMS.json'), 'armed': armed,
            'num_agents': agents, 'body_names': list(bridge.meshes), 'entry': 'visual.mjb',
            'material_gaps': sorted(bridge.material_gaps), 'skin_texture_v_flipped': flip_texture_v}
    meta['texture_color_spaces'] = {**scene_meta.get('texture_color_spaces', {}), **bridge.texture_color_spaces}
    meta['visual_gaps'].extend(meta['material_gaps'])
    (output/'asset.json').write_text(json.dumps(meta, indent=2)+'\n')
    sums = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'SHA256SUMS.json').write_text(json.dumps(sums, indent=2)+'\n')
    return meta


def load_prepared_skin(path, package, poses=None, *, agents=1, armed=False, device='cuda:0',
                       scene_sha256=None, skin_manifest_sha256=None):
    root = Path(path).resolve()
    sums = json.loads((root/'SHA256SUMS.json').read_text())
    if not {'asset.json', 'visual.mjb', 'skin/SHA256SUMS.json', 'skin/skin.json'} <= sums.keys():
        raise ValueError('Prepared skin is missing checksummed metadata')
    for name, expected in sums.items():
        if digest(checked_path(root, name)) != expected:
            raise ValueError(f'Prepared skin checksum mismatch: {name}')
    meta = json.loads((root/'asset.json').read_text())
    if meta['schema'] != 'omega.mjwarp-skin/1' or meta['mujoco_version'] != mujoco.__version__:
        raise ValueError('Prepared skin schema or MuJoCo version mismatch')
    if (meta['policy_sha256'] != package.sha256sum() or meta['policy_body_names'] != package.body_names
            or meta['num_agents'] != agents or meta['armed'] != armed):
        raise ValueError('Prepared skin policy/count/Armed selection mismatch')
    if meta['entry'] != 'visual.mjb' or digest(root/'skin/SHA256SUMS.json') != meta['skin_manifest_sha256']:
        raise ValueError('Prepared skin entry or source manifest mismatch')
    if scene_sha256 is not None and meta.get('scene_sha256') != scene_sha256:
        raise ValueError('Prepared skin scene mismatch')
    if skin_manifest_sha256 is not None and meta['skin_manifest_sha256'] != skin_manifest_sha256:
        raise ValueError('Prepared skin avatar mismatch')
    if poses is None:
        if 'initial_poses.npy' not in sums:
            raise ValueError('Prepared skin initial pose is not checksummed')
        poses = np.load(root/'initial_poses.npy', allow_pickle=False)
    if np.shape(poses) != (agents*len(package.body_names), 7) or not np.isfinite(poses).all():
        raise ValueError('Prepared skin requires finite authoritative body poses')
    model = mujoco.MjModel.from_binary_path(str(root/'visual.mjb'))
    if armed and 'equipment.npz' not in sums:
        raise ValueError('Prepared skin equipment is not checksummed')
    painter = SkinRenderer(load_skin(root/'skin'), package.body_names, package.asset_path, agents, device,
                           armed=armed, equipment_path=root/'equipment.npz' if armed else None)
    bridge = SkinMeshScene()
    painter.log(bridge, poses)
    if list(bridge.meshes) != meta['body_names']:
        raise ValueError('Prepared skin visual body order mismatch')
    bridge.bind_model(model)
    return model, painter, bridge, meta
