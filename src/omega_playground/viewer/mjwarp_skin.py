"""Bridge the existing skin palettes to MJWarp, without Newton or USD.

Topology/materials compile once. Deformed vertices stay on Warp; each mesh BVH
and its conservative scene bounds are updated before drawing. This is currently
an offline replay bridge, not the prepared live-service asset contract.
"""
from pathlib import Path

import mujoco
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation


@wp.kernel
def _local_vertices(points: wp.array[wp.vec3], inverse: wp.transform,
                    output: wp.array[wp.vec3], bounds: wp.array2d[float]):
    i = wp.tid()
    p = wp.transform_point(inverse, points[i])
    output[i] = p
    for axis in range(3):
        wp.atomic_max(bounds, 0, axis, wp.abs(p[axis]))


@wp.kernel
def _set_bound(bounds: wp.array2d[float], mesh_id: int, target: wp.array[wp.vec3]):
    target[mesh_id] = wp.vec3(bounds[0, 0], bounds[0, 1], bounds[0, 2])


@wp.kernel
def _local_normals(normals: wp.array[wp.vec3], inverse: wp.transform, output: wp.array[wp.vec3]):
    i = wp.tid()
    output[i] = wp.normalize(wp.transform_vector(inverse, normals[i]))


class SkinMeshScene:
    def __init__(self, *, flip_texture_v=False, constant_linear_colors=False):
        self.flip_texture_v = flip_texture_v
        self.constant_linear_colors = constant_linear_colors
        self.meshes, self.rigid, self.dynamic = {}, {}, {}
        self.dynamic_normals = {}
        self.material_gaps = set()
        self.texture_color_spaces = {}
        self.renderer = None

    def log_mesh(self, name, points, indices, normals=None, uvs=None, texture=None,
                 color=(1., 1., 1.), roughness=.45, metallic=.2, **kwargs):
        if name in self.meshes or self.renderer is not None:
            raise ValueError('Skin topology must be registered exactly once before compilation')
        self.meshes[name] = {'points': points, 'indices': indices, 'normals': normals, 'uvs': uvs,
                                 'texture': texture, 'color': color, 'roughness': roughness, 'metallic': metallic}

    def bind_skin_material(self, name, root, slot):
        for channel in ('normal', 'roughness', 'metallic'):
            path = Path(root)/f'{channel}_{slot}.png'
            if path.is_file():
                if channel in ('roughness', 'metallic'):
                    from .image_encoding import uniform_scalar_texture
                    value = uniform_scalar_texture(path)
                    if value is not None:
                        self.meshes[name][channel] = value
                        continue
                self.material_gaps.add(f'Textured {channel} is not yet rendered by MJWarp')

    def update_rigid_meshes(self, names, poses):
        for name, pose in zip(names, poses, strict=True):
            self.rigid[name] = np.asarray(pose, np.float32).copy()

    def update_deformed_meshes(self, slices, points, normals):
        for name, start, end in slices:
            self.dynamic[name] = points[start:end]
            if normals is not None:
                self.dynamic_normals[name] = normals[start:end]

    def compile(self, spec):
        for name, part in self.meshes.items():
            points, faces = part['points'].numpy(), part['indices'].numpy().reshape(-1, 3)
            fields = {'name': name, 'uservert': points.ravel(), 'userface': faces.ravel(),
                          'inertia': mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL}
            if part['normals'] is not None:
                fields.update(usernormal=part['normals'].numpy().ravel(), userfacenormal=faces.ravel())
            if part['uvs'] is not None:
                uv = part['uvs'].numpy().copy()
                if self.flip_texture_v:
                    uv[:, 1] = 1.-uv[:, 1]
                fields.update(usertexcoord=uv.ravel(), userfacetexcoord=faces.ravel())
            spec.add_mesh(**fields)
            material = spec.add_material(name=name, rgba=[*part['color'][:3], 1.],
                                         roughness=part['roughness'], metallic=part['metallic'])
            if part['texture']:
                spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(part['texture']))
                material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
                from .image_encoding import declared_image_encoding
                self.texture_color_spaces[name] = declared_image_encoding(part['texture'])
                if self.constant_linear_colors and self.texture_color_spaces[name] == 'auto':
                    from .image_encoding import constant_material_encoding
                    self.texture_color_spaces[name] = constant_material_encoding(
                        part['texture'], part['uvs'].numpy() if part['uvs'] is not None else None)
            spec.worldbody.add_body(name=name).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name, material=name, contype=0, conaffinity=0)
        model = spec.compile()
        self.bind_model(model, validate_initial=True)
        return model

    def bind_model(self, model, *, validate_initial=False):
        """Bind prepared geometry without compiling or assuming the initial pose."""
        self.bindings = {}
        for name, part in self.meshes.items():
            mid = model.mesh(name).id
            start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            rot = Rotation.from_quat(model.mesh_quat[mid, [1, 2, 3, 0]])
            reconstructed = rot.apply(model.mesh_vert[start:start+count])+model.mesh_pos[mid]
            # User-supplied vertices must retain index order, including coincident
            # vertices with different skin weights. Never infer a nearest match.
            if count != len(part['points']):
                raise ValueError('Prepared skin vertex count mismatch')
            if validate_initial:
                np.testing.assert_allclose(reconstructed, part['points'].numpy(), atol=2e-5)
            fstart, fcount = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            np.testing.assert_array_equal(model.mesh_face[fstart:fstart+fcount], part['indices'].numpy().reshape(-1, 3))
            inv_rot = rot.inv()
            inverse = wp.transform(inv_rot.apply(-model.mesh_pos[mid]), inv_rot.as_quat())
            self.bindings[name] = mid, inverse
            if name not in self.rigid:
                # The first log registers the current world-space output. A
                # loaded bundle may have been compiled in a different pose.
                self.dynamic[name] = part['points']
                if part['normals'] is not None:
                    self.dynamic_normals[name] = part['normals']

    def attach(self, renderer):
        if renderer.body_names != list(self.meshes):
            raise ValueError('Skin visual body order mismatch')
        self.renderer = renderer
        self.bounds = wp.zeros((1, 3), dtype=wp.float32, device=renderer.device)
        self.mesh_bvh_ids = renderer.context.mesh_bvh_id.numpy()
        # Rigid shells (A3/H2) and equipment have authored normals too. Their
        # vertices never enter refit(), so upload the normals once in the
        # compiled mesh frame instead of leaving them flat shaded.
        rigid_normals = {name: part['normals'] for name, part in self.meshes.items()
                         if name in self.rigid and part['normals'] is not None}
        all_normals = {**rigid_normals, **self.dynamic_normals}
        if all_normals:
            renderer.enable_smooth_normals()
            for name, normals in all_normals.items():
                mid, inverse = self.bindings[name]
                old_id = int(self.mesh_bvh_ids[mid])
                old = renderer.context.mesh_registry[old_id]
                mesh = wp.Mesh(points=old.points, indices=old.indices,
                               velocities=wp.zeros(len(normals), dtype=wp.vec3, device=renderer.device))
                renderer.context.mesh_registry[mesh.id] = mesh
                wp.launch(_local_normals, dim=len(normals),
                          inputs=[normals, inverse], outputs=[mesh.velocities],
                          device=renderer.device)
                self.mesh_bvh_ids[mid] = mesh.id
                del renderer.context.mesh_registry[old_id]
            renderer.context.mesh_bvh_id.assign(self.mesh_bvh_ids)

    def poses(self):
        if hasattr(self, 'device_poses'):
            return self.device_poses
        result = np.zeros((len(self.meshes), 7), np.float32)
        result[:, 6] = 1
        for index, name in enumerate(self.meshes):
            if name in self.rigid:
                result[index] = self.rigid[name]
        return result

    def refit(self):
        renderer = self.renderer
        if renderer is None:
            raise RuntimeError('Attach MJWarp renderer before updating skin')
        with wp.ScopedDevice(renderer.device):
            for name, points in self.dynamic.items():
                mid, inverse = self.bindings[name]
                mesh = renderer.context.mesh_registry[int(self.mesh_bvh_ids[mid])]
                self.bounds.zero_()
                wp.launch(_local_vertices, dim=len(points), inputs=[points, inverse],
                          outputs=[mesh.points, self.bounds])
                wp.launch(_set_bound, dim=1, inputs=[self.bounds, mid],
                          outputs=[renderer.context.mesh_bounds_size])
                if name in self.dynamic_normals:
                    wp.launch(_local_normals, dim=len(points),
                              inputs=[self.dynamic_normals[name], inverse], outputs=[mesh.velocities])
                mesh.refit()
