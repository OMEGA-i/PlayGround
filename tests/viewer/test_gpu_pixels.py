import warnings

import numpy as np
import pytest
import warp as wp

from omega_playground.viewer.gpu_pixels import Rgb8Readback


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_rgb8_matches_existing_numpy_conversion_bitwise(device):
    wp.init()
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA required')
    # Every half-code boundary and neighbouring floats, plus out-of-range/NaN.
    ties = (np.arange(255, dtype=np.float32)+.5)/255
    values = np.r_[ties, np.nextafter(ties, -np.inf), np.nextafter(ties, np.inf),
                   np.random.default_rng(7).uniform(-.5, 1.5, 3000).astype(np.float32),
                   np.array([np.nan, np.inf, -np.inf], np.float32)]
    values = np.resize(values, (2, 16, 128, 3)).astype(np.float32)
    reader = Rgb8Readback(values.shape[:3], device)
    for pixels in (values, values[..., ::-1].copy()):
        frame = wp.array(pixels, dtype=wp.vec3, device=device)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            expected = (np.clip(pixels, 0, 1)*255).round().astype(np.uint8)
        actual = reader.read(frame)
        np.testing.assert_array_equal(actual, expected)
        assert actual.nbytes == pixels.nbytes//4


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('scale', [2, 3, 4])
def test_supersample_resolves_in_linear_light_on_device(device, scale):
    wp.init()
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA required')
    pixels = np.random.default_rng(13).random((1, 5*scale, 7*scale, 3), dtype=np.float32)
    linear = np.where(pixels <= .04045, pixels/12.92, ((pixels+.055)/1.055)**2.4)
    mean = linear.reshape(1, 5, scale, 7, scale, 3).mean(axis=(2, 4))
    srgb = np.where(mean <= .0031308, mean*12.92, 1.055*mean**(1/2.4)-.055)
    reader = Rgb8Readback((1, 5, 7), device, supersample=scale)
    actual = reader.read(wp.array(pixels, dtype=wp.vec3, device=device))
    expected = (srgb*255).round().astype(np.uint8)
    np.testing.assert_allclose(actual.astype(int), expected.astype(int), atol=1)
    assert actual.shape == (1, 5, 7, 3)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_edge_filter_preserves_flat_regions_and_softens_diagonal(device):
    wp.init()
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA required')
    y, x = np.mgrid[:32, :32]
    values = np.repeat((x > y)[None, ..., None], 3, axis=-1).astype(np.float32)
    reader = Rgb8Readback((1, 32, 32), device, edge_antialias=True)
    actual = reader.read(wp.array(values, dtype=wp.vec3, device=device))
    assert np.any((actual > 0) & (actual < 255))
    np.testing.assert_array_equal(actual[0, 20:25, 1:5], 0)
    np.testing.assert_array_equal(actual[0, 1:5, 20:25], 255)
    flat = np.full_like(values, .4)
    np.testing.assert_array_equal(reader.read(wp.array(flat, dtype=wp.vec3, device=device)), 102)
