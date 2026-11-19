"""Convert released world rotations into native, named 50 Hz tracker references."""
import hashlib
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@lru_cache(maxsize=8)
def _reference_model(xml_path, xml_sha256):
    # Scene/asset changes require a service restart. The XML digest also protects
    # callers that replace the root XML within the same diagnostic process.
    # The cached model is read-only; each conversion owns its MjData.
    return mujoco.MjModel.from_xml_path(xml_path)


def native_reference(path, xml, robot):
    from joystick.resampling import sample_qpos

    xml_path = str(Path(xml).resolve())
    xml_sha256 = hashlib.sha256(Path(xml_path).read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as clip:
        if (clip['embodiment'].item() != robot or clip['fps'].item() != 25
                or clip['xml_sha256'].item() != xml_sha256):
            raise ValueError('Joystick reference identity mismatch')
        names = list(clip['names'])
        joints, rotations, times = clip['joints'], clip['world_body_rotations'], clip['timestamps']
    if (len(times) < 6 or not np.allclose(np.diff(times), .04, atol=1e-8, rtol=0)
            or rotations.shape != (len(times), len(names), 3, 3)
            or joints.shape != rotations.shape[:-1] or not np.isfinite(rotations).all()
            or not np.isfinite(joints).all()):
        raise ValueError('Invalid 25 FPS body transforms or timestamps')
    model = _reference_model(xml_path, xml_sha256)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    bodies = sorted(set(map(int, model.jnt_bodyid)))
    if names != [model.body(i).name for i in bodies] or model.jnt_type[0] != 0 or np.any(model.jnt_type[1:] != 3):
        raise ValueError('Reference skeleton does not match native free-root/hinge model')
    neutral = Rotation.from_matrix(data.xmat.reshape(-1, 3, 3))
    world = {body: Rotation.from_matrix(rotations[:, i]) for i, body in enumerate(bodies)}
    q = np.tile(model.qpos0, (len(times), 1))
    q[:, :3] = joints[:, 0]
    q[:, 3:7] = world[bodies[0]].as_quat()[:, [3, 0, 1, 2]]
    errors = []
    clipped_samples = 0
    for body in bodies[1:]:
        parent = int(model.body_parentid[body])
        while parent not in world and parent:
            parent = int(model.body_parentid[parent])
        rest = neutral[parent].inv() * neutral[body]
        relative = rest.inv() * world[parent].inv() * world[body]
        ids = np.flatnonzero(model.jnt_bodyid == body)
        axes = model.jnt_axis[ids]
        if len(ids) == 1:
            quat = relative.as_quat()
            angles = (2*np.arctan2(quat[:, :3] @ axes[0], quat[:, 3]))[:, None]
        elif len(ids) == 3:
            order = np.argmax(np.abs(axes), axis=1)
            signs = axes[np.arange(3), order]
            if len(set(order)) != 3 or not np.allclose(abs(signs), 1):
                raise ValueError('Unsupported compound joint axes')
            angles = relative.as_euler(''.join('XYZ'[axis] for axis in order)) * signs
        else:
            raise ValueError('Unsupported joint group')
        angles = np.unwrap((angles+np.pi) % (2*np.pi)-np.pi, axis=0)
        slots = model.jnt_qposadr[ids]
        target = q[:, slots] + angles
        limited = model.jnt_limited[ids].astype(bool)
        clamped = target.copy()
        clamped[:, limited] = np.clip(target[:, limited], *model.jnt_range[ids[limited]].T)
        clipped_samples += np.count_nonzero(np.abs(clamped-target) > 1e-8)
        q[:, slots] = clamped
        fitted = Rotation.identity(len(times))
        for local_index, axis in enumerate(axes):
            fitted *= Rotation.from_rotvec((clamped[:,local_index]-model.qpos0[slots[local_index]])[:,None]*axis)
        errors.append((fitted.inv()*relative).magnitude())

    sample_times = times[0] + np.arange(2*len(q)-1)*.02
    state = sample_qpos(q, sample_times, origin=times[0])
    return dict(**state, timestamps=sample_times, fps=50, embodiment=robot,
                projection_error_p95_rad=np.percentile(errors,95),
                clamped_joint_samples=clipped_samples,
                joint_names=[model.joint(i).name for i in range(1, model.njnt)])
