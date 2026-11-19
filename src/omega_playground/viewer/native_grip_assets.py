"""Prepared visual hand variants for SONIC; never modify the physics builder."""
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def asset_path(package):
    digest = hashlib.sha256(package.asset_path.read_bytes()).hexdigest()[:20]
    return Path(__file__).resolve().parents[3] / 'artifacts/equipment' / f'hands-{digest}.npz'


def prepare(package):
    import mujoco
    from scipy.optimize import least_squares

    from omega_playground.viewer.native_skin import native_grip, prepare_grip_mesh

    output = {}
    a3 = package.robot['id'].startswith('agibot_a3')
    if a3:
        source = Path(__file__).resolve().parents[3] / 'artifacts/skins/a3/robot_skin.npz'
        skin = prepare_grip_mesh(np.load(source), 'a3')
        for side, word in [('l', 'left'), ('r', 'right')]:
            hand = skin['names'].tolist().index(word+'_hand')
            bind = skin['bind'][hand]
            points, faces = [], []
            offset = 0
            for slot in range(int(skin['slots'])):
                w, j = skin[f's{slot}_weights'], skin[f's{slot}_joints']
                if not np.all((w*(j == hand)).sum(1) > .9):
                    continue
                p = (skin[f's{slot}_points'] - bind[:3, 3]) @ bind[:3, :3]
                points.append(p)
                faces.append(skin[f's{slot}_indices'].reshape(-1, 3) + offset)
                offset += len(p)
            output[side+'_points'] = np.concatenate(points)
            output[side+'_indices'] = np.concatenate(faces).reshape(-1)
            output[side+'_center'] = native_grip('a3', side)
    else:
        model = mujoco.MjModel.from_xml_path(str(package.asset_path))
        state = mujoco.MjData(model)
        for side, word in [('l', 'left'), ('r', 'right')]:
            wrist = model.body(word+'_wrist_yaw_link').id
            ids = [i for i in range(model.njnt) if (model.joint(i).name or '').startswith(word+'_hand_')]
            coords = model.jnt_qposadr[ids]
            lo, hi = model.jnt_range[ids].T
            bodies = [i for i in range(model.nbody) if (model.body(i).name or '').startswith(word+'_hand_')]
            tips = [model.body(word+'_hand_'+n+'_1_link').id for n in ('index', 'middle')]
            thumb = model.body(word+'_hand_thumb_2_link').id
            # Grip cylinder crosses the knuckles along local Z; left fingers close toward -Y.
            sign = -1 if side == 'l' else 1
            center = np.array([.12, sign*.03, 0.])
            radius = .018 if side == 'l' else .0218
            def points(body, thumb=thumb, sign=sign, wrist=wrist):
                local = np.array([.03, 0., 0.]) if body != thumb else np.array([0., -.026* (-sign), 0.])
                world = state.xpos[body] + state.xmat[body].reshape(3, 3) @ local
                return state.xmat[wrist].reshape(3, 3).T @ (world-state.xpos[wrist])
            def residual(q, coords=coords, tips=tips, center=center, radius=radius, sign=sign, thumb=thumb, points=points):
                state.qpos[coords] = q
                mujoco.mj_forward(model, state)
                result = []
                for body in tips:
                    target = center.copy(); target[0] -= radius+.005
                    target[2] = .0285 if body == tips[0] else -.0285
                    result.extend((points(body)-target)*100)
                target = center + [0., -sign*(radius+.005), 0.]
                result.extend((points(thumb)-target)*100)
                return result
            best = None
            rng = np.random.default_rng(7)
            for _ in range(8):
                fit = least_squares(residual, rng.uniform(lo, hi), bounds=(lo, hi), max_nfev=160)
                if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun): best = fit
            residual(best.x)
            matrices = []
            rw = state.xmat[wrist].reshape(3, 3)
            for body in bodies:
                m = np.eye(4)
                m[:3, :3] = rw.T @ state.xmat[body].reshape(3, 3)
                m[:3, 3] = rw.T @ (state.xpos[body] - state.xpos[wrist])
                matrices.append(m)
            output[side+'_names'] = np.array([model.body(i).name for i in bodies])
            output[side+'_transforms'] = np.array(matrices)
            output[side+'_center'] = center
            output[side+'_fit_error'] = np.linalg.norm(best.fun)/100
    path = asset_path(package)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    return path
