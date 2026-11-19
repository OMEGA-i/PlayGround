"""Conservative encoding identification for offline color textures."""
from PIL import Image


def declared_image_encoding(path):
    with Image.open(path) as image:
        info = image.info
        # ICC may describe a different gamut/transfer function. Do not silently
        # interpret an arbitrary profile or gamma-only declaration as sRGB.
        if not info.get('icc_profile') and info.get('srgb') in (0, 1, 2, 3):
            return 'sRGB'
    return 'auto'


def constant_material_encoding(path, uvs):
    """Validate the audited robot exporter: numeric RGBA -> uniform 2x2 PNG."""
    import numpy as np
    if uvs is None or not np.allclose(uvs, .5, atol=0, rtol=0):
        raise ValueError('Constant robot material requires center UVs')
    with Image.open(path) as image:
        pixels = np.asarray(image)
        if image.mode != 'RGB' or image.size != (2, 2) or not np.all(pixels == pixels[0, 0]):
            raise ValueError('Constant robot material requires a uniform 2x2 RGB image')
        if image.info.get('icc_profile') or 'srgb' in image.info or 'gamma' in image.info:
            raise ValueError('Constant robot material has conflicting color metadata')
    return 'raw'


def uniform_scalar_texture(path):
    """Losslessly fold an untagged uniform grayscale data map into a scalar.

    Return None for spatially varying, colored or transfer-tagged inputs; those
    still require actual texture sampling rather than a guessed material value.
    """
    import numpy as np
    with Image.open(path) as image:
        if image.mode not in ('L', 'RGB') or any(
                key in image.info for key in ('icc_profile', 'srgb', 'gamma')):
            return None
        pixels = np.asarray(image)
        value = pixels.flat[0]
        if not np.all(pixels == value):
            return None
        return float(value)/255.
