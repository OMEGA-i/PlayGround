import numpy as np
import pytest
import warp as wp

from omega_playground.viewer.pbr import direct_brdf


@wp.kernel
def evaluate(rough: wp.array[float], metal: wp.array[float], view: wp.array[wp.vec3],
             light: wp.array[wp.vec3], out: wp.array2d[wp.vec3]):
    i = wp.tid()
    d, s = direct_brdf(wp.vec3(.8, .3, .1), rough[i], metal[i], wp.vec3(0., 0., 1.), view[i], light[i])
    out[i, 0] = d
    out[i, 1] = s


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_metal_diffuse_zero_normal_incidence_and_reciprocity(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    views = np.array([[0, 0, 1]]*3+[[.6, 0, .8], [0, .8, .6], [0, 0, -1]], dtype=np.float32)
    lights = np.array([[0, 0, 1]]*3+[[0, .8, .6], [.6, 0, .8], [0, 0, 1]], dtype=np.float32)
    out = wp.zeros((6, 2), dtype=wp.vec3, device=device)
    wp.launch(evaluate, dim=6, inputs=[wp.array([.5]*6, dtype=float, device=device),
        wp.array([0, 1, .5, .5, .5, 0], dtype=float, device=device),
        wp.array(views, dtype=wp.vec3, device=device), wp.array(lights, dtype=wp.vec3, device=device)], outputs=[out], device=device)
    values = out.numpy()
    assert np.isfinite(values).all() and (values >= 0).all()
    np.testing.assert_allclose(values[1, 0], 0)
    np.testing.assert_allclose(values[0, 0], np.array([.8, .3, .1])*.96/np.pi, rtol=1e-6)
    np.testing.assert_allclose(values[0, 1], .04/(4*np.pi*.5**4), rtol=1e-6)
    np.testing.assert_allclose(values[3]/lights[3, 2], values[4]/lights[4, 2], rtol=1e-6)
    np.testing.assert_allclose(values[5], 0)
