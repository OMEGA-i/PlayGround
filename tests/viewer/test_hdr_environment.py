import numpy as np
import pytest

from omega_playground.viewer.hdr_environment import cubemap_linear, light_bins, read_rgbe, skybox_bytes


def test_rgbe_runs_decode_known_linear_values_and_reject_truncation(tmp_path):
    path = tmp_path/'small.hdr'
    header = b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n'
    # R literal 8 pixels, G/B/exponent repeated runs.
    row = bytes([2, 2, 0, 8, 8, 0, 32, 64, 96, 128, 160, 192, 224,
                 136, 64, 136, 32, 136, 129])
    path.write_bytes(header+row)
    rgb = read_rgbe(path)
    np.testing.assert_array_equal(rgb[0, :, 0], np.arange(8)*.25)
    np.testing.assert_array_equal(rgb[0, :, 1], [.5]*8)
    np.testing.assert_array_equal(rgb[0, :, 2], [.25]*8)
    path.write_bytes(header+row[:-1])
    with pytest.raises(ValueError, match='truncated'):
        read_rgbe(path)


def test_constant_dome_conserves_integral_and_rotation():
    image = np.ones((64, 128, 3))
    bins, report = light_bins(image, np.eye(3))
    np.testing.assert_allclose(report['mean_rgb'], [1, 1, 1], atol=1e-12)
    np.testing.assert_allclose(np.sum([b['diffuse'] for b in bins], axis=0), [4]*3, atol=1e-12)
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotated, _ = light_bins(image, rotation)
    np.testing.assert_allclose([b['incoming'] for b in rotated],
                               np.array([b['incoming'] for b in bins])@rotation.T, atol=1e-12)
    # Finite-bin quadrature approximates the cosine convolution of an isotropic dome.
    incoming = np.array([b['incoming'] for b in bins])
    energies = np.array([b['diffuse'] for b in bins])
    for normal in np.concatenate((np.eye(3), -np.eye(3))):
        irradiance = np.maximum(incoming@normal, 0)@energies
        np.testing.assert_allclose(irradiance, [1]*3, atol=.15)


def test_bright_north_pole_has_positive_y_direction():
    image = np.zeros((64, 128, 3))
    image[:8] = [1, .5, .25]
    bins, _ = light_bins(image, np.eye(3))
    assert all(b['incoming'][1] > .9 for b in bins)
    assert all(b['diffuse'][0] > b['diffuse'][1] > b['diffuse'][2] for b in bins)


def test_cube_faces_follow_world_axes_and_dome_rotation():
    # Match source HDR resolution; pole sampling clamps at the top texel center.
    latitude = np.pi/2-(np.arange(1024)+.5)*np.pi/1024
    longitude = np.pi-(np.arange(2048)+.5)*2*np.pi/2048
    xyz = np.stack(np.broadcast_arrays(np.cos(latitude[:, None])*np.sin(longitude),
                   np.sin(latitude[:, None]), np.cos(latitude[:, None])*np.cos(longitude)), axis=-1)
    rgb = (xyz+1)/2
    axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0], [0, 1, 0]])
    for rotation in (np.eye(3), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])):
        cube = cubemap_linear(rgb, rotation, 9).reshape(6, 9, 9, 3)
        np.testing.assert_allclose(cube[:, 4, 4], (axes@rotation+1)/2, atol=.001)
    data = np.frombuffer(skybox_bytes(np.zeros_like(rgb), np.eye(3), 0, size=9), dtype=np.uint8)
    assert data.shape == (6*9*9*3,)
    assert not data.any()
