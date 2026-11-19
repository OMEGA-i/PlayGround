"""Ray-footprint texture filtering, independently of scene lighting."""
import numpy as np
import pytest
import warp as wp
from omega_playground.viewer.mjwarp_pbr_render import uv_differential, texture_lod


@wp.kernel
def footprint(out: wp.array[wp.vec2]):
    out[0] = uv_differential(wp.vec3(2., 0., 0.), wp.vec3(0., 4., 0.),
                             wp.vec2(1., 0.), wp.vec2(0., 1.), wp.vec3(.5, 1., 0.))
    out[1] = wp.vec2(texture_lod(wp.vec2(.25, 0.), wp.vec2(0., .125), 32, 32),
                     texture_lod(wp.vec2(.001), wp.vec2(.001), 32, 32))
    out[2] = uv_differential(wp.vec3(0.), wp.vec3(0.), wp.vec2(1.), wp.vec2(1.), wp.vec3(1.))


def test_metric_uv_gradient_and_lod():
    out = wp.empty(3, dtype=wp.vec2, device='cpu')
    wp.launch(footprint, dim=1, outputs=[out], device='cpu')
    np.testing.assert_allclose(out.numpy(), [[.25, .25], [3, 0], [0, 0]], atol=1e-6)


@wp.kernel
def checker_samples(tex: wp.Texture2D, out: wp.array[wp.vec4]):
    i = wp.tid()
    uv = wp.vec2((float(i)+.5)/32., .5/32.)
    out[i] = wp.texture_sample(tex, uv, dtype=wp.vec4,
        lod=texture_lod(wp.vec2(.25, 0.), wp.vec2(0., .25), tex.width, tex.height))


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_minified_checker_converges_to_area_mean(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    checker = (np.indices((32, 32)).sum(axis=0) % 2).astype(np.float32)
    pixels = np.repeat(checker[..., None], 4, axis=2)
    pixels[..., 3] = 1.
    texture = wp.Texture2D(pixels, device=device, num_mip_levels=0)
    out = wp.empty(32, dtype=wp.vec4, device=device)
    wp.launch(checker_samples, dim=32, inputs=[texture], outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy()[:, :3], .5, atol=1e-6)
    np.testing.assert_allclose(out.numpy()[:, 3], 1.)


@pytest.mark.parametrize('pbr', [False, True])
def test_native_and_pbr_renderers_use_mips_on_cuda(pbr):
    if not wp.is_cuda_available():
        pytest.skip('CUDA renderer required')
    import mujoco
    from omega_playground.viewer.mjwarp import MjWarpPoseRenderer
    spec = mujoco.MjSpec()
    spec.add_texture(name='checker', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=32, height=32)
    material = spec.add_material(name='surface', roughness=.6, metallic=0.)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'checker'
    spec.worldbody.add_body(name='robot').add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[.5,.5,.5], material='surface')
    spec.worldbody.add_camera(name='camera')
    renderer = MjWarpPoseRenderer(spec.compile(), ['robot'], width=64, height=64,
                                 pbr=pbr, capture_graph=False)
    renderer.look_at([2., -2., 1.], [0., 0., 0.])
    pixels = renderer.render(np.array([[0,0,0,0,0,0,1]], np.float32)).numpy()
    assert np.isfinite(pixels).all() and pixels.std() > .01
    assert renderer.linear_texture_registry[0]._num_mip_levels > 1
