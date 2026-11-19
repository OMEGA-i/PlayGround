import numpy as np
import pytest
import warp as wp
from omega_playground.viewer.environment_radiance import upload_radiance
from omega_playground.viewer.environment_importance import EnvironmentDistribution, prepare_distribution, mis_environment


@wp.kernel
def integrate(pixels: wp.array2d[wp.vec3], distribution: EnvironmentDistribution, out: wp.array[wp.vec3]):
    i = wp.tid()
    rotation = wp.mat33(1., 0., 0., 0., 1., 0., 0., 0., 1.)
    rough = .045 if i == 0 else 1.
    out[i] = mis_environment(pixels, distribution, rotation, wp.vec3(1.), wp.vec3(1.),
                             rough, 1., wp.vec3(0., 0., 1.), wp.vec3(0., 0., 1.), 1024)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_mis_preserves_white_furnace_energy(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    rgb = np.ones((32, 64, 3))*2
    distribution = prepare_distribution(rgb, 1024, device)
    np.testing.assert_allclose(distribution.pdf.numpy(), 1/(4*np.pi), rtol=1e-6)
    out = wp.empty(2, dtype=wp.vec3, device=device)
    wp.launch(integrate, dim=2, inputs=[upload_radiance(rgb, device), distribution], outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy()[0], 2., atol=.01)
    np.testing.assert_allclose(out.numpy()[1], 2*(1-np.log(2)), atol=.01)


def test_distribution_normalized_and_samples_bright_region():
    rgb = np.ones((32, 64, 3))
    rgb[16, 32] = 1e6
    distribution = prepare_distribution(rgb, 1024, 'cpu')
    area = (np.cos(np.arange(32)*np.pi/32)-np.cos((np.arange(32)+1)*np.pi/32))*2*np.pi/64
    pdf = distribution.pdf.numpy()
    np.testing.assert_allclose((pdf*area[:, None]).sum(), 1., atol=1e-6)
    assert (pdf > 0).all()
    samples = distribution.samples.numpy()
    np.testing.assert_allclose(np.linalg.norm(samples[:, :3], axis=-1), 1., atol=1e-6)
    assert (samples[:, 2] > .95).sum() > 900
    assert np.isfinite(samples).all() and (samples[:, 3] > 0).all()
