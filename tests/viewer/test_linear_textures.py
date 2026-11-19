import numpy as np
import pytest
import warp as wp
from omega_playground.viewer.linear_textures import decode


@wp.kernel
def midpoint(texture: wp.Texture2D, out: wp.array[wp.vec4]):
    out[0] = wp.texture_sample(texture, wp.vec2(.5, .5), dtype=wp.vec4)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('channels', [1, 2, 3, 4])
def test_decode_preserves_alpha_and_linear_values(device, channels):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    values = np.array([128]*channels, np.uint8)
    out = wp.empty((1, 1, 4), dtype=float, device=device)
    wp.launch(decode, dim=(1, 1), inputs=[1, 0, channels, True, wp.array(values, device=device)], outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy()[0, 0, :3], .2158605, atol=1e-7)
    np.testing.assert_allclose(out.numpy()[0, 0, 3], 128/255 if channels in (2, 4) else 1)


def test_gpu_filters_decoded_texels_not_encoded_average():
    if not wp.is_cuda_available():
        pytest.skip('CUDA texture sampling required')
    with wp.ScopedDevice('cuda:0'):
        pixels = wp.empty((1, 2, 4), dtype=float)
        wp.launch(decode, dim=(2, 1), inputs=[2, 0, 3, True,
            wp.array([0, 0, 0, 255, 255, 255], dtype=wp.uint8)], outputs=[pixels])
        texture = wp.Texture2D(pixels, filter_mode=wp.TextureFilterMode.LINEAR)
        out = wp.empty(1, dtype=wp.vec4)
        wp.launch(midpoint, dim=1, inputs=[texture], outputs=[out])
        np.testing.assert_allclose(out.numpy()[0], [.5, .5, .5, 1], atol=1e-6)


def test_bound_texture_requires_resolved_metadata_but_unused_texture_does_not():
    import mujoco
    from omega_playground.viewer.linear_textures import rgb_encodings
    s = mujoco.MjSpec()
    for name in ('used', 'unused'):
        s.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=2, height=2)
        material = s.add_material(name=name)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1, 0, 0], material='used')
    m = s.compile()
    for declarations in ({}, {'used': 'auto'}):
        with pytest.raises(ValueError, match='requires resolved'):
            rgb_encodings(m, declarations)
    assert rgb_encodings(m, {'used': 'sRGB'}) == {m.tex('used').id: 'sRGB'}


@wp.kernel
def sample_repeat_seam(texture: wp.Texture2D, out: wp.array[wp.vec4]):
    out[0] = wp.texture_sample(texture, wp.vec2(0., .5), dtype=wp.vec4)


def test_uploaded_linear_texture_filters_across_repeat_seam():
    if not wp.is_cuda_available():
        pytest.skip('CUDA texture sampling required')
    import mujoco
    from types import SimpleNamespace
    from omega_playground.viewer.linear_textures import install_linear_textures
    spec = mujoco.MjSpec()
    spec.add_texture(name='seam', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=2, height=2)
    model = spec.compile()
    pixels = model.tex_data.reshape(2, 2, -1)
    pixels[:, 0] = 0
    pixels[:, 1] = 255
    context = SimpleNamespace()
    registry = install_linear_textures(model, context, {0: 'raw'}, 'cuda:0')
    with wp.ScopedDevice('cuda:0'):
        out = wp.empty(1, dtype=wp.vec4)
        wp.launch(sample_repeat_seam, dim=1, inputs=[registry[0]], outputs=[out])
        np.testing.assert_allclose(out.numpy()[0, :3], [.5]*3, atol=1e-6)


@wp.kernel
def sample_outside(texture: wp.Texture2D, uv: wp.vec2, out: wp.array[wp.vec4]):
    out[0] = wp.texture_sample(texture, uv, dtype=wp.vec4)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('mode,expected', [('repeat', 0.), ('clamp', 1.), ('mirror', 1.), ('black', 0.)])
def test_uploaded_texture_preserves_per_axis_addressing(axis, mode, expected):
    if not wp.is_cuda_available():
        pytest.skip('CUDA texture sampling required')
    import mujoco
    from types import SimpleNamespace
    from omega_playground.viewer.linear_textures import install_linear_textures
    spec = mujoco.MjSpec()
    spec.add_texture(name='address', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=2, height=2)
    model = spec.compile()
    pixels = model.tex_data.reshape(2, 2, -1)
    pixels[:] = 0
    if axis == 0:
        pixels[:, 1] = 255
    else:
        pixels[1, :] = 255
    modes = ['repeat', 'repeat']
    modes[axis] = mode
    registry = install_linear_textures(model, SimpleNamespace(), {0: 'raw'}, 'cuda:0', {'address': modes})
    uv = [.25, .25]
    uv[axis] = 1.25
    with wp.ScopedDevice('cuda:0'):
        out = wp.empty(1, dtype=wp.vec4)
        wp.launch(sample_outside, dim=1, inputs=[registry[0], wp.vec2(*uv)], outputs=[out])
        np.testing.assert_allclose(out.numpy()[0, :3], [expected]*3, atol=1e-6)
