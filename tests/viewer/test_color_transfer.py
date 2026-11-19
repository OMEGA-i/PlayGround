import numpy as np
import pytest
import warp as wp
from omega_playground.viewer.color_transfer import linear_to_srgb, srgb_to_linear


@wp.kernel
def transfer(x: wp.array[float], encoded: wp.array[float], decoded: wp.array[float]):
    i = wp.tid()
    encoded[i] = linear_to_srgb(x[i])
    decoded[i] = srgb_to_linear(encoded[i])


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_srgb_reference_values_and_roundtrip(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    values = np.array([0, .001, .0031308, .018, .18, .5, 1], np.float32)
    x = wp.array(values, device=device)
    y = wp.empty_like(x)
    z = wp.empty_like(x)
    wp.launch(transfer, dim=len(values), inputs=[x], outputs=[y, z], device=device)
    expected = np.where(values <= .0031308, 12.92*values, 1.055*values**(1/2.4)-.055)
    np.testing.assert_allclose(y.numpy(), expected, atol=1e-7)
    np.testing.assert_allclose(z.numpy(), values, atol=2e-7)
    assert abs(y.numpy()[4]-.461356) < 1e-6
