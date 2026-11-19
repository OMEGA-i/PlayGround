"""Offline/diagnostic extraction of linear USD dome radiance for MJWarp."""
import hashlib
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdLux

from .hdr_environment import read_rgbe, light_bins


def load_dome(entry):
    stage = Usd.Stage.Open(str(entry))
    lights = []
    domes = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI) or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        api = UsdLux.LightAPI(prim)
        intensity = float(api.GetIntensityAttr().Get())*2.**float(api.GetExposureAttr().Get())
        lights.append(intensity)
        if prim.IsA(UsdLux.DomeLight):
            domes.append((prim, api, intensity))
    if len(domes) != 1:
        raise ValueError('HDR integration currently requires exactly one visible dome')
    prim, api, intensity = domes[0]
    if api.GetEnableColorTemperatureAttr().Get():
        raise ValueError('Dome color temperature requires conversion')
    dome = UsdLux.DomeLight(prim)
    if dome.GetTextureFormatAttr().Get() not in ('latlong', 'automatic'):
        raise ValueError('Expected latlong dome')
    texture = dome.GetTextureFileAttr().Get()
    if not texture or not texture.resolvedPath:
        raise ValueError('Unresolved HDR dome texture')
    path = Path(texture.resolvedPath)
    image = read_rgbe(path)
    if image.shape[1] != 2*image.shape[0]:
        raise ValueError('Expected 2:1 HDR latlong')
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rotation = np.asarray(matrix)[:3, :3]  # USD row-vector local->world = column world->local.
    _, report = light_bins(image, rotation.T)
    scale = 1./max(max(lights), 1e-12)
    gain = np.asarray(api.GetColorAttr().Get())*intensity*scale*float(api.GetSpecularAttr().Get())/max(report['mean_luminance'], 1e-12)
    return image, rotation, gain, {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'world_to_dome': rotation.tolist(), 'gain': gain.tolist(),
        'normalization': 'matches existing relative USD intensity and mean HDR luminance; not photometric',
        'occlusion': False, 'integration': 'GGX NDF importance sampling, single scattering'}
