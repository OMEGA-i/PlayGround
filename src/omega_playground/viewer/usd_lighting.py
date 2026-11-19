"""Offline USD lighting approximation for MJWarp's non-photometric shader."""
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLux


def add_usd_lighting(spec, entry):
    stage = Usd.Stage.Open(str(entry))
    lights, gaps = [], set()
    skybox = None
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if prim.GetTypeName() not in ('DistantLight', 'DomeLight'):
            gaps.add(f'Unsupported USD light: {prim.GetTypeName()}')
            continue
        api = UsdLux.LightAPI(prim)
        if api.GetEnableColorTemperatureAttr().Get():
            gaps.add('USD light color temperature is not yet applied')
        intensity = float(api.GetIntensityAttr().Get()) * 2.**float(api.GetExposureAttr().Get())
        color = np.asarray(api.GetColorAttr().Get(), dtype=float)
        if not np.isfinite(intensity) or intensity < 0 or not np.isfinite(color).all() or (color < 0).any():
            raise ValueError(f'Invalid USD light intensity/color: {prim.GetPath()}')
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        direction = np.asarray(matrix.TransformDir(Gf.Vec3d(0, 0, -1)))
        if np.linalg.norm(direction) < 1e-10:
            raise ValueError('Degenerate USD light transform')
        lights.append({'path': str(prim.GetPath()), 'type': prim.GetTypeName(),
                       'intensity': intensity, 'color': color.tolist(),
                       'direction': (direction/np.linalg.norm(direction)).tolist(),
                       'diffuse': float(api.GetDiffuseAttr().Get()),
                       'specular': float(api.GetSpecularAttr().Get())})
        if prim.IsA(UsdLux.DomeLight):
            dome = UsdLux.DomeLight(prim)
            texture = dome.GetTextureFileAttr().Get()
            if texture and texture.path:
                if dome.GetTextureFormatAttr().Get() not in ('latlong', 'automatic'):
                    raise ValueError('Only latlong dome textures are supported')
                from .hdr_environment import light_bins, read_rgbe, skybox_bytes
                source = Path(texture.resolvedPath)
                if not texture.resolvedPath or not source.is_file():
                    raise ValueError(f'Unresolved dome texture: {texture.path}')
                image = read_rgbe(source)
                if image.shape[1] != image.shape[0]*2:
                    raise ValueError('Expected 2:1 latlong dome texture')
                bins, report = light_bins(image, np.asarray(matrix)[:3, :3].T)
                lights[-1]['environment'] = {**report, 'source': texture.path,
                    'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'bins': bins}
                if skybox is None:
                    skybox = skybox_bytes(image, np.asarray(matrix)[:3, :3].T,
                                          report['mean_luminance'], color)
                    lights[-1]['environment']['background'] = {
                        'face_size': 256, 'mapping': 'MJWarp cube faces from USD world directions',
                        'display': 'mean-luminance normalized Reinhard then sRGB, 8 bit'}
                else:
                    gaps.add('Multiple HDR domes: background uses the first visible dome')
    if not lights:
        return {'lights': [], 'visual_gaps': sorted(gaps), 'imported': False}
    # MuJoCo diffuse/ambient are unitless coefficients. Preserve authored light
    # ratios, not physical USD radiance; never advertise exposure parity.
    scale = 1./max(max(light['intensity'] for light in lights), 1e-12)
    ambient = np.zeros(3)
    suns = []
    environment = []
    for light in lights:
        if light['type'] == 'DomeLight':
            gain = np.asarray(light['color'])*light['intensity']*scale*light['diffuse']
            if 'environment' in light:
                environment.extend((sample, gain) for sample in light['environment']['bins'])
                gaps.add('HDR diffuse uses 16 unoccluded directional bins; specular IBL not yet rendered')
            else:
                ambient += gain
        else:
            suns.append(light)
            gaps.add('USD sun angular size/soft shadows are not yet rendered')
    for light in list(spec.worldbody.lights):
        spec.delete(light)
    if skybox is not None:
        for texture in list(spec.textures):
            if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                spec.delete(texture)
        texture = spec.add_texture(name='scene_hdr_sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                                   width=256, height=1536, nchannel=3)
        texture.data = skybox
        gaps.add('HDR background tone-mapped to 8-bit sRGB; no shared photometric exposure pipeline')
    spec.visual.headlight.active = 0
    for index, light in enumerate(suns or [None]):
        rgb = np.asarray(light['color'])*light['intensity']*scale if light else np.zeros(3)
        spec.worldbody.add_light(name=f'scene_light_{index}',
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            dir=light['direction'] if light else [0, 0, -1],
            diffuse=rgb*light['diffuse'] if light else rgb,
            specular=rgb*light['specular'] if light else rgb,
            ambient=ambient/max(len(suns), 1), castshadow=bool(light))
    for index, (sample, gain) in enumerate(environment):
        spec.worldbody.add_light(name=f'environment_light_{index}',
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            dir=-np.asarray(sample['incoming']), diffuse=np.asarray(sample['diffuse'])*gain,
            specular=[0, 0, 0], ambient=[0, 0, 0], castshadow=False)
    gaps.add('USD intensities normalized to unitless MJWarp coefficients; no photometric exposure parity')
    return {'lights': lights, 'intensity_scale': scale, 'imported': True,
            'visual_gaps': sorted(gaps)}
