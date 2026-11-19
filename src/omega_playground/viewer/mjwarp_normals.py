"""Renderer-local MJWarp adapter for authored smooth normals.

Visual-only Warp meshes store normals in the optional velocity channel; these
meshes never participate in physics. No upstream module globals are mutated.
"""
from types import FunctionType

import numpy as np
import warp as wp
from mujoco_warp._src import render as upstream
from mujoco_warp._src.ray import ray_mesh_with_bvh


def install_compiled_normals(model, context, mesh_ids, device):
    """Upload rigid mesh corner normals without changing triangle/UV order.

    Split only on (vertex, normal) pairs: hard edges remain hard, and shared
    smooth vertices stay shared. Never use this topology for deforming skins.
    """
    ids = context.mesh_bvh_id.numpy()
    changed = 0
    for mid in sorted(set(map(int, mesh_ids))):
        fa, fc = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        va, na = int(model.mesh_vertadr[mid]), int(model.mesh_normaladr[mid])
        faces = model.mesh_face[fa:fa+fc]
        normal_indices = model.mesh_facenormal[fa:fa+fc]
        if not fc or np.any(normal_indices < 0):
            continue
        pairs, inverse = np.unique(np.c_[faces.ravel(), normal_indices.ravel()],
                                   axis=0, return_inverse=True)
        old_id = int(ids[mid])
        mesh = wp.Mesh(
            points=wp.array(model.mesh_vert[va+pairs[:, 0]], dtype=wp.vec3, device=device),
            indices=wp.array(inverse.astype(np.int32), dtype=wp.int32, device=device),
            velocities=wp.array(model.mesh_normal[na+pairs[:, 1]], dtype=wp.vec3, device=device))
        context.mesh_registry[mesh.id] = mesh
        ids[mid] = mesh.id
        del context.mesh_registry[old_id]
        changed += 1
    context.mesh_bvh_id.assign(ids)
    return changed


@wp.func
def smooth_mesh_hit(mesh_bvh_id: wp.array[wp.uint64], mesh_geom_id: int,
                    pos: wp.vec3, mat: wp.mat33, pnt: wp.vec3, vec: wp.vec3,
                    max_t: float, cull_backfaces: bool):
    t, normal, u, v, face, mesh = ray_mesh_with_bvh(
        mesh_bvh_id, mesh_geom_id, pos, mat, pnt, vec, max_t, cull_backfaces)
    if face >= 0:
        authored = wp.mesh_eval_velocity(mesh_bvh_id[mesh_geom_id], face, u, v)
        if wp.length_sq(authored) > 1.0e-12:
            smooth = wp.normalize(mat @ authored)
            if wp.dot(smooth, normal) < 0.0:
                smooth = -smooth
            normal = smooth
        # Double-sided surfaces must shade the visible side. Keeping the back
        # normal points lighting/shadow rays through thin foliage and shells.
        if not cull_backfaces and wp.dot(normal, vec) > 0.0:
            normal = -normal
    return t, normal, u, v, face, mesh


def _with_globals(function, **overrides):
    namespace = dict(function.__globals__)
    namespace.update(overrides)
    specialized = FunctionType(function.__code__, namespace, function.__name__,
                               function.__defaults__, function.__closure__)
    specialized.__annotations__ = function.__annotations__.copy()
    return specialized


def smooth_render_function(module=upstream):
    """Specialize primary intersections, retaining upstream shadow traversal."""
    original = module._make_cast_ray
    primary = _with_globals(original, ray_mesh_with_bvh=smooth_mesh_hit)

    def make_cast_ray(geom_ray_types, first_hit=False, **kwargs):
        return (original if first_hit else primary)(geom_ray_types, first_hit=first_hit, **kwargs)

    return _with_globals(module.render.__wrapped__, _make_cast_ray=make_cast_ray)
