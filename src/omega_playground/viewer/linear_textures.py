"""Explicit RGB texture decoding before GPU bilinear filtering."""
import mujoco
import numpy as np
import warp as wp

from .color_transfer import srgb_to_linear


@wp.kernel
def decode(width: int, address: int, channels: int, encoded: bool,
           source: wp.array[wp.uint8], output: wp.array3d[float]):
    x, y = wp.tid()
    offset = address+(y*width+x)*channels
    for c in range(3):
        channel = c if channels >= 3 else 0
        value = float(source[offset+channel])/255.0
        if encoded:
            value = srgb_to_linear(value)
        output[y, x, c] = value
    alpha = float(1.0)
    if channels == 2 or channels == 4:
        alpha = float(source[offset+channels-1])/255.0
    output[y, x, 3] = alpha


def rgb_encodings(model, declarations):
    """Require declared encoding for textures referenced by bound materials."""
    materials = np.unique(model.geom_matid[model.geom_matid >= 0])
    ids = np.unique(model.mat_texid[materials, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
    result = {}
    for texture_id in ids[ids >= 0]:
        name = model.tex(int(texture_id)).name
        value = declarations.get(name)
        if value not in ('sRGB', 'raw'):
            raise ValueError(f'RGB texture {name} requires resolved sRGB/raw metadata, got {value}')
        result[int(texture_id)] = value
    return result


def texture_address_modes(declaration):
    modes = {'repeat': wp.TextureAddressMode.WRAP, 'clamp': wp.TextureAddressMode.CLAMP,
             'mirror': wp.TextureAddressMode.MIRROR, 'black': wp.TextureAddressMode.BORDER}
    if not isinstance(declaration, (list, tuple)) or len(declaration) != 2 or any(
            mode not in modes for mode in declaration):
        raise ValueError(f'Invalid S/T texture addressing: {declaration}')
    return tuple(modes[mode] for mode in declaration)


def install_linear_textures(model, context, encodings, device, wrap_modes=None):
    from mujoco_warp._src.render_util import create_warp_texture
    # Context owns the new textures through this returned registry. Upload once,
    # not per frame; texture filtering subsequently operates on linear values.
    with wp.ScopedDevice(device):
        source = wp.array(model.tex_data, dtype=wp.uint8)
        registry = []
        for i in range(model.ntex):
            if i not in encodings:
                registry.append(create_warp_texture(model, i))
                continue
            width, height, channels = int(model.tex_width[i]), int(model.tex_height[i]), int(model.tex_nchannel[i])
            if channels not in (1, 2, 3, 4):
                raise ValueError('RGB texture requires 1–4 channels')
            pixels = wp.empty((height, width, 4), dtype=float)
            wp.launch(decode, dim=(width, height), inputs=[width, int(model.tex_adr[i]), channels,
                encodings[i] == 'sRGB', source], outputs=[pixels])
            # Address raw UVs and the bilinear footprint together, including seams.
            addressing = texture_address_modes((wrap_modes or {}).get(
                model.tex(i).name, ('repeat', 'repeat')))
            registry.append(wp.Texture2D(pixels, filter_mode=wp.TextureFilterMode.LINEAR,
                                         address_mode=addressing, num_mip_levels=0))
        context.textures = wp.array(registry, dtype=wp.Texture2D)
    return registry
