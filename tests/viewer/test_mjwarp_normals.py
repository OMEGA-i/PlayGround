import numpy as np
import warp as wp

from omega_playground.viewer.mjwarp_normals import smooth_mesh_hit, smooth_render_function, upstream


@wp.kernel
def sample_hit(ids: wp.array[wp.uint64], output: wp.array[wp.vec3], distance: wp.array[float]):
    t, normal, _u, _v, _face, _mesh = smooth_mesh_hit(
        ids, 0, wp.vec3(0.0), wp.identity(n=3, dtype=float),
        wp.vec3(0.25, 0.25, 1.0), wp.vec3(0.0, 0.0, -1.0), 5.0, True)
    output[0] = normal
    distance[0] = t


def test_authored_normals_interpolate_without_changing_hit():
    points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device='cpu')
    indices = wp.array([0, 1, 2], dtype=wp.int32, device='cpu')
    normals = np.array([[0, 0, 1], [.6, 0, .8], [0, .6, .8]], dtype=np.float32)
    for smooth in (False, True):
        mesh = wp.Mesh(points=points, indices=indices,
                       velocities=wp.array(normals, dtype=wp.vec3, device='cpu') if smooth else None)
        output = wp.zeros(1, dtype=wp.vec3, device='cpu')
        distance = wp.zeros(1, dtype=float, device='cpu')
        wp.launch(sample_hit, dim=1, inputs=[wp.array([mesh.id], dtype=wp.uint64, device='cpu')],
                  outputs=[output, distance], device='cpu')
        expected = normals[0]*.5+normals[1]*.25+normals[2]*.25 if smooth else np.array([0, 0, 1])
        np.testing.assert_allclose(output.numpy()[0], expected/np.linalg.norm(expected), atol=1e-6)
        np.testing.assert_allclose(distance.numpy(), [1.])


def test_renderer_adapter_does_not_patch_upstream_globals():
    original = upstream._make_cast_ray
    render = smooth_render_function()
    assert upstream._make_cast_ray is original
    assert render.__globals__['_make_cast_ray'] is not original
    assert upstream.render.__wrapped__.__globals__['_make_cast_ray'] is original


def test_smooth_pbr_adapter_keeps_material_arguments_and_module_isolation():
    import inspect
    from omega_playground.viewer import mjwarp_pbr_render
    original = mjwarp_pbr_render._make_cast_ray
    render = smooth_render_function(mjwarp_pbr_render)
    assert inspect.signature(render) == inspect.signature(mjwarp_pbr_render.render)
    assert mjwarp_pbr_render._make_cast_ray is original
    assert render.__globals__['_make_cast_ray'] is not original
    assert render.__globals__['_make_compute_lighting'] is mjwarp_pbr_render._make_compute_lighting


def test_compiled_corner_normals_preserve_hard_edges_and_triangle_order():
    from types import SimpleNamespace
    from omega_playground.viewer.mjwarp_normals import install_compiled_normals
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    faces = np.array([[0, 1, 2], [0, 3, 1]], np.int32)
    normals = np.array([[0, 0, 1], [0, 1, 0]], np.float32)
    old = wp.Mesh(points=wp.array(points, dtype=wp.vec3, device='cpu'),
                  indices=wp.array(faces.ravel(), dtype=wp.int32, device='cpu'))
    context = SimpleNamespace(mesh_bvh_id=wp.array([old.id], dtype=wp.uint64, device='cpu'),
                              mesh_registry={old.id: old})
    model = SimpleNamespace(mesh_faceadr=[0], mesh_facenum=[2], mesh_vertadr=[0],
                            mesh_normaladr=[0], mesh_vert=points, mesh_face=faces,
                            mesh_normal=normals, mesh_facenormal=np.array([[0,0,0],[1,1,1]]))
    assert install_compiled_normals(model, context, [0, 0], 'cpu') == 1
    mesh = context.mesh_registry[int(context.mesh_bvh_id.numpy()[0])]
    indices = mesh.indices.numpy().reshape(-1, 3)
    np.testing.assert_array_equal(mesh.points.numpy()[indices], points[faces])
    np.testing.assert_array_equal(mesh.velocities.numpy()[indices], normals[model.mesh_facenormal])
    assert len(mesh.points) == 6  # Split both shared vertices at the hard edge.


@wp.kernel
def sample_backface(ids: wp.array[wp.uint64], output: wp.array[wp.vec3]):
    t, normal, _u, _v, _face, _mesh = smooth_mesh_hit(
        ids, 0, wp.vec3(0.0), wp.identity(n=3, dtype=float),
        wp.vec3(.25, .25, -1.), wp.vec3(0., 0., 1.), 5., False)
    output[0] = normal


def test_double_sided_surface_shades_toward_viewer():
    mesh = wp.Mesh(points=wp.array([[0,0,0], [1,0,0], [0,1,0]], dtype=wp.vec3, device='cpu'),
                   indices=wp.array([0,1,2], dtype=wp.int32, device='cpu'),
                   velocities=wp.array([[0,0,1]]*3, dtype=wp.vec3, device='cpu'))
    output = wp.empty(1, dtype=wp.vec3, device='cpu')
    wp.launch(sample_backface, dim=1, inputs=[wp.array([mesh.id], dtype=wp.uint64, device='cpu')],
              outputs=[output], device='cpu')
    np.testing.assert_allclose(output.numpy()[0], [0,0,-1], atol=1e-6)
