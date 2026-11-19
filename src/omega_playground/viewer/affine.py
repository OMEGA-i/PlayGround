"""Separate mesh forward transforms for bounds from inverse-transpose ray transforms."""
import mujoco
import numpy as np


def rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def static_mesh_affines(model, instances):
    ids, positions, forward, ray = [], [], [], []
    for instance in instances:
        geom = model.geom(instance['geom']).id
        if model.geom_bodyid[geom] != 0 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError('Affine instances must be static world mesh geoms')
        mesh = model.geom_dataid[geom]
        scale = np.asarray(instance['scale'], dtype=float)
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(np.abs(scale) < 1e-10):
            raise ValueError('Invalid instance scale')
        basis = rotation(instance['quat'])
        canonical = rotation(model.mesh_quat[mesh])
        # MuJoCo recenters and rotates mesh vertices during compilation.
        # Restore that local frame before the source USD instance transform.
        positions.append(np.asarray(instance['pos'])+basis@(scale*model.mesh_pos[mesh]))
        forward.append(basis@np.diag(scale)@canonical)
        ray.append(basis@np.diag(1/scale)@canonical)
        ids.append(geom)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate static visual instance')
    return (np.asarray(ids, np.int32), np.asarray(positions, np.float32),
            np.asarray(forward, np.float32), np.asarray(ray, np.float32))
