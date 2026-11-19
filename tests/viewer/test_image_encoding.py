from PIL import Image, PngImagePlugin
from omega_playground.viewer.image_encoding import declared_image_encoding


def test_srgb_chunk_is_authoritative_but_gamma_alone_is_not(tmp_path):
    image = Image.new('RGB', (2, 2))
    for name, chunk, expected in [('srgb', ('sRGB', b'\x00'), 'sRGB'),
                                  ('gamma', ('gAMA', (45455).to_bytes(4, 'big')), 'auto')]:
        metadata = PngImagePlugin.PngInfo()
        metadata.add(chunk[0].encode(), chunk[1])
        path = tmp_path/f'{name}.png'
        image.save(path, pnginfo=metadata)
        assert declared_image_encoding(path) == expected
    path = tmp_path/'unmarked.png'
    image.save(path)
    assert declared_image_encoding(path) == 'auto'



def test_constant_robot_material_requires_audited_shape_and_uvs(tmp_path):
    import numpy as np
    import pytest
    from omega_playground.viewer.image_encoding import constant_material_encoding
    path = tmp_path/'constant.png'
    Image.new('RGB', (2, 2), (128, 64, 32)).save(path)
    assert constant_material_encoding(path, np.full((4, 2), .5)) == 'raw'
    with pytest.raises(ValueError, match='center UVs'):
        constant_material_encoding(path, np.zeros((4, 2)))
    image = Image.open(path)
    image.putpixel((0, 0), (255, 0, 0))
    image.save(path)
    with pytest.raises(ValueError, match='uniform'):
        constant_material_encoding(path, np.full((4, 2), .5))


def test_uniform_data_texture_is_not_averaged_or_gamma_decoded(tmp_path):
    from omega_playground.viewer.image_encoding import uniform_scalar_texture
    from PIL import Image, PngImagePlugin
    p = tmp_path/'rough.png'
    Image.new('RGB', (2, 2), (127, 127, 127)).save(p)
    assert uniform_scalar_texture(p) == 127/255
    img = Image.new('RGB', (2, 2), (127, 127, 127))
    img.putpixel((0, 0), (128, 128, 128))
    img.save(p)
    assert uniform_scalar_texture(p) is None
    Image.new('RGB', (2, 2), (127, 128, 127)).save(p)
    assert uniform_scalar_texture(p) is None
    meta = PngImagePlugin.PngInfo()
    meta.add(b'sRGB', bytes([0]))
    Image.new('RGB', (2, 2), (127, 127, 127)).save(p, pnginfo=meta)
    assert uniform_scalar_texture(p) is None
