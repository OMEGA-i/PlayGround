"""Authored opacity cutouts shared by primary and shadow rays."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import warp as wp


@wp.struct
class Cutouts:
    slot: wp.array[int]
    textures: wp.array[wp.Texture2D]
    threshold: wp.array[float]
    face_base: wp.array[int]
    uv_base: wp.array[int]
    face_uv: wp.array[wp.vec3i]
    uv: wp.array[wp.vec2]


def load_cutouts(model, context, directory, declarations, device):
    from .linear_textures import texture_address_modes
    cutouts = Cutouts()
    if not declarations:
        return cutouts, []
    slots = np.full(model.ngeom, -1, dtype=np.int32)
    textures, thresholds = [], []
    for material, spec in declarations.items():
        mat_id = model.mat(material).id
        texture_id = int(model.mat_texid[mat_id, 1])
        path = Path(directory)/'textures'/spec['file']
        if texture_id < 0 or not model.tex(texture_id).name.startswith('campus_tex_' + path.stem[:16] + '_'):
            raise ValueError(f'Opacity binding differs from prepared RGB texture on {material}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != path.stem:
            raise ValueError(f'Opacity texture checksum mismatch: {path}')
        with Image.open(path) as img:
            alpha = np.asarray(img.convert('RGBA'), dtype=np.float32)[:, :, 3] / 255
        slots[model.geom_matid == mat_id] = len(textures)
        modes = texture_address_modes(spec['wrap'])
        textures.append(wp.Texture2D(wp.array(np.ascontiguousarray(alpha), device=device),
                                     address_mode=modes, normalized_coords=True))
        thresholds.append(spec['threshold'])
    cutouts.slot = wp.array(slots, dtype=int, device=device)
    cutouts.textures = wp.array(textures, dtype=wp.Texture2D, device=device)
    cutouts.threshold = wp.array(thresholds, dtype=float, device=device)
    cutouts.face_base = wp.array(model.mesh_faceadr, dtype=int, device=device)
    cutouts.uv_base = context.mesh_texcoord_offsets
    cutouts.face_uv = context.mesh_facetexcoord
    cutouts.uv = context.mesh_texcoord
    return cutouts, textures


@wp.func
def opaque(mask: Cutouts, slot: int, mesh: int, face: int, u: float, v: float):
    corners = mask.face_uv[mask.face_base[mesh] + face]
    base = mask.uv_base[mesh]
    st = u * mask.uv[base + corners[0]] + v * mask.uv[base + corners[1]] + (1.0-u-v) * mask.uv[base + corners[2]]
    return wp.texture_sample(mask.textures[slot], st, dtype=float) >= mask.threshold[slot]
