"""CUDA/BVH rendering of authoritative body poses; no physics stepping or RTX.

The prepared MuJoCo model describes visual geometry only. Its body names map
explicitly to the flattened agent-major snapshot, independent of PhysX DOFs.
"""
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


@wp.kernel
def _set_camera(position: wp.vec3, rotation: wp.mat33,
                positions: wp.array2d[wp.vec3], rotations: wp.array2d[wp.mat33]):
    positions[0, 0] = position
    rotations[0, 0] = rotation


@wp.kernel
def _place_geometries(poses: wp.array[wp.transform], slots: wp.array[int],
                      local_pos: wp.array[wp.vec3], local_rot: wp.array[wp.quat],
                      geom_pos: wp.array2d[wp.vec3], geom_rot: wp.array2d[wp.mat33]):
    g = wp.tid()
    slot = slots[g]
    if slot >= 0:
        body = poses[slot]
        geom_pos[0, g] = wp.transform_point(body, local_pos[g])
        rotation = wp.mul(wp.transform_get_rotation(body), local_rot[g])
        geom_rot[0, g] = wp.quat_to_matrix(rotation)


@wp.kernel
def _place_static(ids: wp.array[int], positions: wp.array[wp.vec3], matrices: wp.array[wp.mat33],
                  geom_pos: wp.array2d[wp.vec3], geom_mat: wp.array2d[wp.mat33]):
    i = wp.tid()
    geom_pos[0, ids[i]] = positions[i]
    geom_mat[0, ids[i]] = matrices[i]


class MjWarpPoseRenderer:
    def __init__(self, model, body_names, *, width=640, height=480, device='cuda:0', shadows=True, static_instances=(), capture_graph=True, backface_culling=True, fast_math=True, pbr=False, linear_output=False, texture_color_spaces=None, texture_wrap_modes=None, textures=True, skin_meshes=False, cutout_materials=None, visual_directory=None, face_varying_normals=None):
        if len(set(body_names)) != len(body_names):
            raise ValueError('Render snapshot body names must be unique')
        if model.ncam < 1:
            raise ValueError('Prepared render model requires a camera')
        if linear_output:
            if not pbr:
                raise ValueError('Linear output requires the experimental PBR shader')
            from .linear_textures import rgb_encodings
            encodings = rgb_encodings(model, texture_color_spaces or {})
        self.linear_output = bool(linear_output)
        from .visual_geometry import render_model
        model = render_model(model)
        # USD's V=0 is the bottom row. MuJoCo stores the imported PNG top row first.
        scene_meshes = {int(model.geom(row['geom']).dataid[0]) for row in static_instances}
        for mesh_id in scene_meshes:
            start, count = model.mesh_texcoordadr[mesh_id], model.mesh_texcoordnum[mesh_id]
            if count:
                uv = model.mesh_texcoord[start:start+count, 1]
                uv[:] = 1.0 - uv
        # MuJoCo can replace authored corner normal indices with vertex indices.
        # Recover the USD triangle corner order on the independent rendering model.
        for name, order in (face_varying_normals or {}).items():
            mesh_id = model.mesh(name).id
            first, count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
            if model.mesh_normalnum[mesh_id] != 3 * count:
                raise ValueError(f'Unexpected face-varying normal count on {name}')
            model.mesh_facenormal[first:first+count] = np.arange(3*count).reshape(-1,3)[:,order]
        if static_instances and not linear_output and any(
                model.light(i).name.startswith('environment_light_') for i in range(model.nlight)):
            # The SDR shader sums the sun and all HDR dome bins directly.
            # Expose a horizontal diffuse white patch within the display range.
            irradiance = model.light_ambient + model.light_diffuse * np.maximum(-model.light_dir[:,2:], 0)
            exposure = 1.0 / max(1.0, float(irradiance.sum(axis=0).max()))
            model.light_diffuse[:] *= exposure
            model.light_specular[:] *= exposure
            model.light_ambient[:] *= exposure
        self.device, self.body_names = device, list(body_names)
        self.capture_graph, self.graph = capture_graph, None
        self.render_function = mjw.render
        self.pbr = pbr
        self.environment = None
        self.environment_rotation = None
        self.environment_gain = None
        self.environment_samples = 0
        self.environment_distribution = None
        self.width, self.height = width, height
        names = {name: i for i, name in enumerate(body_names)}
        slots = []
        for body in model.geom_bodyid:
            name = model.body(int(body)).name
            if body != 0 and name not in names:
                raise ValueError(f'Render body missing from pose contract: {name}')
            slots.append(names.get(name, -1) if body else -1)
        with wp.ScopedDevice(device):
            initial = mujoco.MjData(model)
            mujoco.mj_kinematics(model, initial)
            mujoco.mj_camlight(model, initial)
            self.model = mjw.put_model(model)
            self.data = mjw.put_data(model, initial)
            from .mjwarp_pbr_render import render as filtered_render
            # The filtered primary-ray path is shared by native and skinned
            # avatars. Negative scalar values preserve the legacy Phong branch
            # when PBR was not selected; mip filtering is not a PBR-only feature.
            self.roughness = wp.array(model.mat_roughness if pbr else np.full(model.nmat, -1.), dtype=float)
            self.metallic = wp.array(model.mat_metallic if pbr else np.full(model.nmat, -1.), dtype=float)
            self.render_function = lambda m, d, rc: filtered_render(m, d, rc, self.roughness, self.metallic, self.linear_output, self.environment, self.environment_rotation, self.environment_gain, self.environment_samples, self.environment_distribution, self.cutouts)

            self.context = mjw.create_render_context(model, nworld=1, cam_res=(width, height),
                                                     # look_at() positions camera 0 for each requested view.
                                                     # Other cameras embedded in the MJCF have no image consumer.
                                                     cam_active=[i == 0 for i in range(model.ncam)],
                                                     render_rgb=True, render_depth=True,
                                                     use_textures=textures, use_shadows=shadows,
                                                     render_skybox=True,
                                                     enable_backface_culling=backface_culling, use_fast_math=fast_math)
            from .opacity import load_cutouts
            masks, self.cutout_textures = load_cutouts(model, self.context, visual_directory, cutout_materials, device)
            self.cutouts = masks if self.cutout_textures else None
            from .linear_textures import install_linear_textures
            if not linear_output:
                encodings = {i: 'raw' for i in range(model.ntex)
                             if model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_2D}
            self.linear_texture_registry = install_linear_textures(
                model, self.context, encodings, device, texture_wrap_modes)
            # MJWarp bounds are centered on the mesh origin, which need not
            # equal its AABB midpoint after MuJoCo's inertia-frame compilation.
            # Half-range alone can clip off-center mesh tips. Symmetric bounds
            # about the actual origin conservatively include every vertex.
            bounds = self.context.mesh_bounds_size.numpy()
            for mesh in range(model.nmesh):
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                bounds[mesh] = np.abs(model.mesh_vert[start:start+count]).max(axis=0)
            self.context.mesh_bounds_size.assign(bounds)
            from .mjwarp_normals import install_compiled_normals
            mesh_geoms = model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
            if skin_meshes:
                # SkinMeshScene owns mutable body-mesh topology and normals.
                mesh_geoms &= model.geom_bodyid == 0
            if install_compiled_normals(model, self.context, model.geom_dataid[mesh_geoms], device):
                self.enable_smooth_normals()
            self.slots = wp.array(slots, dtype=wp.int32)
            self.local_pos = wp.array(model.geom_pos, dtype=wp.vec3)
            self.local_rot = wp.array(model.geom_quat[:, [1, 2, 3, 0]], dtype=wp.quat)
            self.poses = wp.empty(len(body_names), dtype=wp.transform)
            self.rgb = wp.empty((1, height, width), dtype=wp.vec3)
            self.static = None
            if static_instances:
                from omega_playground.viewer.affine import static_mesh_affines
                ids, positions, forward, ray = static_mesh_affines(model, static_instances)
                self.static = (wp.array(ids, dtype=wp.int32), wp.array(positions, dtype=wp.vec3),
                               wp.array(forward, dtype=wp.mat33), wp.array(ray, dtype=wp.mat33))

    def enable_smooth_normals(self, enabled=True):
        from .mjwarp_normals import smooth_render_function
        from . import mjwarp_pbr_render
        shader = smooth_render_function(mjwarp_pbr_render) if enabled else mjwarp_pbr_render.render
        self.render_function = lambda m, d, rc: shader(m, d, rc, self.roughness, self.metallic, self.linear_output, self.environment, self.environment_rotation, self.environment_gain, self.environment_samples, self.environment_distribution, self.cutouts)
        self.graph = None  # Previously captured draws use the previous shader.

    def set_environment(self, rgb, world_to_dome, gain=(1., 1., 1.), samples=32, importance_sampling=True):
        """Experimental unoccluded HDR specular integration; upload only once."""
        if not self.pbr or not self.linear_output:
            raise ValueError('HDR environment requires linear PBR output')
        rotation = np.asarray(world_to_dome, dtype=np.float32)
        gain = np.asarray(gain, dtype=np.float32)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
            raise ValueError('Expected proper world-to-dome rotation')
        if gain.shape != (3,) or not np.isfinite(gain).all() or (gain < 0).any():
            raise ValueError('Expected finite nonnegative environment gain')
        if not isinstance(samples, int) or not 1 <= samples <= 1024:
            raise ValueError('Environment samples must be an integer in 1..1024')
        from .environment_radiance import upload_radiance
        self.environment = upload_radiance(rgb, self.device)
        self.environment_rotation = wp.mat33(*rotation.ravel())
        self.environment_gain = wp.vec3(*gain)
        self.environment_samples = samples
        from .environment_importance import prepare_distribution
        self.environment_distribution = prepare_distribution(rgb, samples, self.device) if importance_sampling else None
        self.graph = None

    def set_prepared_environment(self, arrays):
        """Upload validated offline arrays without constructing a distribution."""
        if not self.pbr or not self.linear_output:
            raise ValueError('HDR environment requires linear PBR output')
        from .environment_importance import EnvironmentDistribution
        self.environment = wp.array(arrays['radiance'], dtype=wp.vec3, device=self.device)
        self.environment_rotation = wp.mat33(*arrays['world_to_dome'].ravel())
        self.environment_gain = wp.vec3(*arrays['gain'])
        distribution = EnvironmentDistribution()
        distribution.pdf = wp.array(arrays['pdf'], dtype=float, device=self.device)
        distribution.samples = wp.array(arrays['samples'], dtype=wp.vec4, device=self.device)
        self.environment_distribution = distribution
        self.environment_samples = len(arrays['samples'])
        self.graph = None

    def look_at(self, eye, target):
        """Move the first camera in world space without altering physical poses."""
        eye, target = np.asarray(eye, dtype=np.float32), np.asarray(target, dtype=np.float32)
        if eye.shape != (3,) or target.shape != (3,) or not np.isfinite([eye, target]).all():
            raise ValueError('Camera eye and target must be finite XYZ vectors')
        z = eye-target
        x = np.cross([0., 0., 1.], z)
        if np.linalg.norm(z) < 1e-6 or np.linalg.norm(x) < 1e-6:
            raise ValueError('Camera direction must be nonzero and not parallel to world up')
        z /= np.linalg.norm(z)
        x /= np.linalg.norm(x)
        rotation = np.stack((x, np.cross(z, x), z), axis=1)
        wp.launch(_set_camera, dim=1,
                  inputs=[wp.vec3(*eye), wp.mat33(*rotation.ravel())],
                  outputs=[self.data.cam_xpos, self.data.cam_xmat], device=self.device)

    def render(self, body_q):
        if isinstance(body_q, wp.array):
            if (body_q.dtype != wp.transform or body_q.shape != (len(self.body_names),)
                    or body_q.device != wp.get_device(self.device)):
                raise ValueError('Device render poses must match the prepared transform layout')
            # Internal prepared GPU skin output: values never cross the host.
            values = body_q
        else:
            values = np.asarray(body_q, dtype=np.float32)
            if values.shape != (len(self.body_names), 7) or not np.isfinite(values).all():
                raise ValueError('Expected finite world XYZ/XYZW pose for each render body')
            norms = np.linalg.norm(values[:, 3:], axis=-1)
            if not np.allclose(norms, 1., atol=1e-3):
                raise ValueError('Render body rotations must be unit quaternions')
        with wp.ScopedDevice(self.device):
            if isinstance(values, wp.array):
                wp.copy(self.poses, values)
            else:
                self.poses.assign(values)
            if self.capture_graph:
                if self.graph is None:
                    self._draw()  # Compile kernels before CUDA stream capture.
                    with wp.ScopedCapture(device=self.device) as capture:
                        self._draw()
                    self.graph = capture.graph
                wp.capture_launch(self.graph)
            else:
                self._draw()
        return self.rgb

    def _draw(self):
        wp.launch(_place_geometries, dim=len(self.slots),
                  inputs=[self.poses, self.slots, self.local_pos, self.local_rot],
                  outputs=[self.data.geom_xpos, self.data.geom_xmat])
        if self.static:
            ids, positions, forward, ray = self.static
            wp.launch(_place_static, dim=len(ids), inputs=[ids, positions, forward],
                      outputs=[self.data.geom_xpos, self.data.geom_xmat])
        mjw.refit_bvh(self.model, self.data, self.context)
        if self.static:
            # Mesh ray queries map with transpose(mat); use inverse transpose
            # here, while BVH bounds above require the forward transform.
            wp.launch(_place_static, dim=len(ids), inputs=[ids, positions, ray],
                      outputs=[self.data.geom_xpos, self.data.geom_xmat])
        self.render_function(self.model, self.data, self.context)
        mjw.get_rgb(self.context, 0, self.rgb)
