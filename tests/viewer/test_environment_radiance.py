import numpy as np
import pytest
import warp as wp
from omega_playground.viewer.environment_radiance import sample_radiance, upload_radiance


@wp.kernel
def samples(pixels: wp.array2d[wp.vec3], directions: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = sample_radiance(pixels, directions[i])


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_hdr_sampling_seam_poles_and_linear_filter(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    rgb = np.zeros((2, 4, 3), np.float32)
    rgb[0] = np.array([2, 4, 8, 16])[:, None]
    rgb[1] = rgb[0]*2
    directions = wp.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0],
                           [1e-6, 0, -1], [-1e-6, 0, -1]], dtype=wp.vec3, device=device)
    out = wp.empty(6, dtype=wp.vec3, device=device)
    wp.launch(samples, dim=6, inputs=[upload_radiance(rgb, device), directions], outputs=[out], device=device)
    values = out.numpy()
    np.testing.assert_allclose(values[:4], np.repeat([[9], [13.5], [6], [12]], 3, axis=1), atol=1e-5)
    np.testing.assert_allclose(values[4:6], 13.5, atol=3e-5)
    assert values.max() > 1  # HDR energy must survive upload and filtering.


@pytest.mark.parametrize('bad', [np.zeros((0, 2, 3)), np.ones((2, 2, 4)),
                                np.full((2, 2, 3), np.nan), -np.ones((2, 2, 3))])
def test_reject_invalid_radiance(bad):
    with pytest.raises(ValueError):
        upload_radiance(bad, 'cpu')


@wp.kernel
def integrated(pixels: wp.array2d[wp.vec3], out: wp.array[wp.vec3]):
    from_rotation = wp.mat33(1., 0., 0., 0., 1., 0., 0., 0., 1.)
    i = wp.tid()
    rough = .045 if i == 0 else 1.0
    out[i] = specular_environment(pixels, from_rotation, wp.vec3(1.), wp.vec3(1.),
                                 rough, 1., wp.vec3(0., 0., 1.), wp.vec3(0., 0., 1.), 1024)


from omega_playground.viewer.environment_radiance import specular_environment


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_white_furnace_specular_integral(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    out = wp.empty(2, dtype=wp.vec3, device=device)
    wp.launch(integrated, dim=2, inputs=[upload_radiance(np.ones((4, 8, 3))*2, device)], outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy()[0], 2., atol=.001)
    # At NoV=1, alpha=1, F=1: hemispherical integral = 1-ln(2).
    np.testing.assert_allclose(out.numpy()[1], 2*(1-np.log(2)), atol=.001)
