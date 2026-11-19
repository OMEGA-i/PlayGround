from types import SimpleNamespace

import mujoco
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from omega_playground.viewer.mjwarp_skin import SkinMeshScene


def test_deformation_preserves_duplicate_indices_and_updates_bounds():
    scene = SkinMeshScene()
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], np.float32)
    vertices += [12, -8, 2]
    faces = np.array([0, 2, 1, 4, 1, 3, 0, 3, 2, 1, 2, 3], np.int32)
    scene.log_mesh('test', wp.array(vertices, dtype=wp.vec3, device='cpu'),
                   wp.array(faces, dtype=wp.int32, device='cpu'))
    model = scene.compile(mujoco.MjSpec())
    mesh = wp.Mesh(points=wp.array(model.mesh_vert, dtype=wp.vec3, device='cpu'),
                   indices=wp.array(model.mesh_face.ravel(), dtype=wp.int32, device='cpu'))
    context = SimpleNamespace(mesh_registry={mesh.id: mesh},
        mesh_bvh_id=wp.array([mesh.id], dtype=wp.uint64, device='cpu'),
        mesh_bounds_size=wp.zeros(1, dtype=wp.vec3, device='cpu'))
    scene.attach(SimpleNamespace(body_names=['test'], device='cpu', context=context))
    moved = vertices.copy()
    moved[4] += [0, 0, 3]  # Coincident bind vertices can deform independently.
    points = wp.array(moved, dtype=wp.vec3, device='cpu')
    scene.update_deformed_meshes([('test', 0, 5)], points, None)
    scene.refit()
    restored = Rotation.from_quat(model.mesh_quat[0, [1, 2, 3, 0]]).apply(mesh.points.numpy())+model.mesh_pos[0]
    np.testing.assert_allclose(restored, moved, atol=2e-5)
    np.testing.assert_allclose(context.mesh_bounds_size.numpy()[0], np.abs(mesh.points.numpy()).max(axis=0))
    np.testing.assert_allclose(scene.poses(), [[0, 0, 0, 0, 0, 0, 1]])


def test_explicit_v_flip_preserves_source_uv_and_face_indices():
    uv = np.array([[.1, .2], [.3, .4], [.5, .6], [.7, .8]], np.float32)
    points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=wp.vec3, device='cpu')
    indices = wp.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=wp.int32, device='cpu')
    source = wp.array(uv, dtype=wp.vec2, device='cpu')
    bridge = SkinMeshScene(flip_texture_v=True)
    bridge.log_mesh('skin', points, indices, uvs=source)
    model = bridge.compile(mujoco.MjSpec())
    np.testing.assert_array_equal(source.numpy(), uv)
    np.testing.assert_allclose(model.mesh_texcoord, np.c_[uv[:, 0], 1.-uv[:, 1]])
    np.testing.assert_array_equal(model.mesh_facetexcoord, indices.numpy().reshape(-1, 3))


def test_constant_pbr_maps_reach_compiled_material(tmp_path):
    from PIL import Image
    bridge = SkinMeshScene()
    points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=wp.vec3, device='cpu')
    indices = wp.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=wp.int32, device='cpu')
    bridge.log_mesh('skin', points, indices)
    Image.new('RGB', (2, 2), (64, 64, 64)).save(tmp_path/'roughness_0.png')
    Image.new('RGB', (2, 2), (255, 255, 255)).save(tmp_path/'metallic_0.png')
    Image.new('RGB', (2, 2), (128, 128, 255)).save(tmp_path/'normal_0.png')
    bridge.bind_skin_material('skin', tmp_path, 0)
    model = bridge.compile(mujoco.MjSpec())
    np.testing.assert_allclose(model.mat_roughness, [64/255])
    np.testing.assert_allclose(model.mat_metallic, [1.])
    assert bridge.material_gaps == {'Textured normal is not yet rendered by MJWarp'}


def test_rigid_shell_normals_uploaded_in_compiled_mesh_frame():
    scene = SkinMeshScene()
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, .5]], np.float32)
    vertices += [3, -2, 1]
    faces = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], np.int32)
    normals = np.array([[0, 0, 1], [.6, 0, .8], [0, .6, .8], [1, 0, 0]], np.float32)
    scene.log_mesh('shell', wp.array(vertices, dtype=wp.vec3, device='cpu'),
                   wp.array(faces, dtype=wp.int32, device='cpu'),
                   normals=wp.array(normals, dtype=wp.vec3, device='cpu'))
    scene.update_rigid_meshes(['shell'], [[0, 0, 0, 0, 0, 0, 1]])
    model = scene.compile(mujoco.MjSpec())
    mesh = wp.Mesh(points=wp.array(model.mesh_vert, dtype=wp.vec3, device='cpu'),
                   indices=wp.array(model.mesh_face.ravel(), dtype=wp.int32, device='cpu'))
    context = SimpleNamespace(mesh_registry={mesh.id: mesh},
        mesh_bvh_id=wp.array([mesh.id], dtype=wp.uint64, device='cpu'),
        mesh_bounds_size=wp.zeros(1, dtype=wp.vec3, device='cpu'))
    enabled = []
    scene.attach(SimpleNamespace(body_names=['shell'], device='cpu', context=context,
                                 enable_smooth_normals=lambda: enabled.append(True)))
    installed = context.mesh_registry[int(context.mesh_bvh_id.numpy()[0])]
    rotation = Rotation.from_quat(model.mesh_quat[0, [1, 2, 3, 0]])
    np.testing.assert_allclose(rotation.apply(installed.velocities.numpy()), normals, atol=1e-6)
    np.testing.assert_array_equal(installed.points.numpy(), mesh.points.numpy())
    np.testing.assert_array_equal(installed.indices.numpy(), mesh.indices.numpy())
    assert enabled == [True]
    assert not scene.dynamic_normals and not scene.dynamic
    scene.refit()  # No deformation is needed to retain rigid normals.
    np.testing.assert_allclose(rotation.apply(installed.velocities.numpy()), normals, atol=1e-6)
