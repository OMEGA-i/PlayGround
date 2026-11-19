"""Execute one physical SuperMimic task without the shared web workbench."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from omega_playground.recording.execution import Cancellation, PhysicalTrajectory
from xml.etree import ElementTree as ET

import numpy as np
import torch

from omega_playground.control.tensor_web_player import TensorWebPlayer
from omega_playground.map.navigation import CampusNavigation
from omega_playground.ops.scenes import SCENES
from omega_playground.map.preview import inspect_path
from omega_playground.policy.base import RobotBatchView
from omega_playground.policy.supermimic.chunk import ProprioHistory
from omega_playground.policy.supermimic.heading import HeadingFrame, place_from_proprio
from omega_playground.policy.supermimic.live_client import JoystickClient
from omega_playground.policy.supermimic.release import tracker_checkpoint
from omega_playground.policy.supermimic.rollout import TrackingSession
from omega_playground.policy.supermimic.provenance import runtime, file_identity, repository

ROOT = Path(__file__).resolve().parents[2]
CONTACT_SHA = {'obstacle': '754213e8a512ddffb988202916a58f6b2e9257ef9c5671f711e3ba48a409a8af',
    'campus': 'a1f8f8087b1462f1cc977b3c0084df7946a296890f5fff4291bad44815965e4f'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def collision_asset(task, shared):
    """Resolve a pinned collision asset; scene coordinates retain their release anchor."""
    scene = task['scene']
    baseline = ROOT.parent / f'{scene}-full-contacts-v2.xml'
    if sha(baseline) != CONTACT_SHA[scene]:
        raise ValueError('Scene collision asset differs from accepted release')
    candidate = task.get('physical_collision')
    if candidate is None:
        return baseline
    relative = Path(candidate['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('physical_collision.path must be relative to shared root')
    path = shared / relative
    if sha(path) != candidate['sha256']:
        raise ValueError('Pinned physical collision asset hash mismatch')
    return path


def record_task(task, output, cancellation=None, *, planner_factory=JoystickClient, session_factory=TrackingSession, navigation=None):
    cancellation = cancellation or Cancellation()
    import mujoco
    output.mkdir(parents=True, exist_ok=False)
    shared = Path(os.environ.get('OMEGA_SHARED_ROOT', Path.home() / 'omega-assets'))
    robot, scene = task['embodiment'], task['scene']
    if robot not in ('g1', 'a3', 'h2', 'sih2300') or scene not in SCENES:
        raise ValueError('Unknown released embodiment or scene')
    xml = (shared / 'joystick/assets/sih2300/robot.xml' if robot == 'sih2300'
        else shared / 'supermimic/assets/robots' / robot / 'robot.xml')
    contact = collision_asset(task, shared)
    nav = navigation if navigation is not None else CampusNavigation(ROOT / 'artifacts/maps' / SCENES[scene][1])
    route = np.asarray(task['route'], dtype=float)
    spec = json.loads((ROOT / 'configs/scenes' / SCENES[scene][2]).read_text())
    if not np.allclose(route[0], spec['route'][0]):
        raise ValueError('Route spawn differs from collision release origin')
    clearance = float(task.get('planning_clearance_m', task.get('clearance_m', .6)))
    for key in ('scene_sha256', 'collision_sha256'):
        if key in task and task[key] != nav.meta[key]:
            raise ValueError(f'Task {key} does not match navigation asset')
    validation = inspect_path(nav, route.tolist(), clearance)
    if not validation['valid']:
        raise ValueError(validation['message'])
    if task.get('require_detour', scene == 'obstacle') and nav.line_valid(route[0], route[-1], clearance):
        raise ValueError('Avoid-obstacles task must require a detour around blocked direct path')
    seed = int(task.get('seed', 20261107))
    initial_lift = float(task.get('initial_root_lift_m', 0.))
    if not math.isfinite(initial_lift) or not 0 <= initial_lift <= .05:
        raise ValueError('initial_root_lift_m must be between zero and 0.05 metres')
    measured_heading = task.get('measured_heading', True)
    if type(measured_heading) is not bool:
        raise ValueError('measured_heading must be a boolean')
    replan_steps = task.get('replan_control_steps')
    if replan_steps is not None and (type(replan_steps) is not int or replan_steps < 1):
        raise ValueError('replan_control_steps must be a positive integer')
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['OMEGA_SUPERMIMIC_CONTACTS'] = str(contact)
    planner = planner_factory(robot, xml)
    try:
        session = session_factory(robot, tracker_checkpoint(robot, shared), xml)
        if task.get('tracker_sha256', session.checkpoint_sha256) != session.checkpoint_sha256:
            raise ValueError('Task requires a different frozen Tracker release')
        initial_heading = math.atan2(*(route[1] - route[0])[::-1])
        initial_reference = yield ('plan', planner, 0., 0., None)
        if initial_lift:
            initial_reference = dict(initial_reference, qpos=initial_reference['qpos'].copy())
            initial_reference['qpos'][:, 2] += initial_lift
        session.append(HeadingFrame([0., 0.], initial_heading).place(initial_reference, initial_heading))
        state = session.env.state()
        n = len(session.names[0])
        size = 13 + 2*n
        ground = ET.parse(ROOT.parent / f'{scene}-full-contacts-v2.xml').find('worldbody/geom')
        offset = torch.tensor([*route[0], -float(ground.get('pos', '0 0 0').split()[2])], device=session.device)
        scene_from_tracker = np.eye(4)
        scene_from_tracker[:3, 3] = offset.cpu().numpy()
        def physical():
            return RobotBatchView((state[:, :3] + offset)[:, None], state[:, [4,5,6,3]][:, None],
                state[:, 7:10][:, None], state[:, 10:13][:, None], state[:, 13:13+n],
                state[:, 13+n:13+2*n], offset[2:3])
        spec.update(route=route.tolist(), task_mode='drawn_path')
        spec['reference'].update(preserve_path=False, nominal_speed=float(task.get('speed_m_s', task.get('speed', .45))))
        player = TensorWebPlayer(spec, session.device)
        smooth_validation = inspect_path(nav, player.reference.xy.tolist(), clearance)
        if not smooth_validation['valid']:
            raise ValueError('Smoothed execution route: ' + smooth_validation['message'])
        write_json(output/'planned_route.json', dict(route=player.reference.xy.tolist(), clearance_m=clearance,
            reference=player.reference.describe()))
        player.reset(torch.tensor([0], device=session.device), physical())
        player.configure_locomotion('walk', float(task.get('speed_m_s', task.get('speed', .45))), 0.)
        if task.get('start_mode') == 'aligned':
            player.start_aligned(physical(), 0.)
        model = session.env.cpu_model
        jids = [model.joint(name).id for name in session.names[0]]
        embodiment = dict(id=robot, joint_names=session.names[0], model_sha256=sha(xml),
            joint_axes=model.jnt_axis[jids].tolist(), joint_positions=model.jnt_pos[jids].tolist(),
            joint_limits=model.jnt_range[jids].tolist(), joint_body_ids=model.jnt_bodyid[jids].tolist(),
            body_names=[model.body(i).name for i in range(model.nbody)], body_parent_ids=model.body_parentid.tolist(),
            body_mass=model.body_mass.tolist(), body_inertia=model.body_inertia.tolist(),
            body_inertial_position=model.body_ipos.tolist(), body_inertial_quaternion_wxyz=model.body_iquat.tolist(),
            pd_kp=model.actuator_gainprm[:, 0].tolist(), pd_kd=(-model.actuator_biasprm[:, 2]).tolist())
        if getattr(session, 'model_path', None) is not None:
            os.link(session.model_path, output/'physics.mjb')
        else:
            mujoco.mj_saveModel(model, str(output/'physics.mjb'), None)
        checkpoint_stat = Path(session.checkpoint_identity['path']).stat()
        if (checkpoint_stat.st_mtime_ns != session.checkpoint_mtime_ns
                or checkpoint_stat.st_size != session.checkpoint_identity['size_bytes']):
            raise ValueError('Tracker checkpoint changed after verified loading')
        manifest = dict(schema='omega.playground.execution/1', sdg_task_uid=str(uuid.uuid4()),
            task=dict(description=task.get('description', 'walk from start to goal, avoid obstacles'), parameters=task, result='running'),
            embodiment=embodiment, scene=dict(id=scene, sha256=nav.meta['scene_sha256'], collision_sha256=nav.meta['collision_sha256'], physical_collision_sha256=sha(contact)),
            scene_from_tracker=scene_from_tracker.tolist(), episode_start_step=0, physics_hz=200, control_hz=50,
            camera_hz=25, seed=seed, versions=dict(playground=repository(ROOT)["commit"], tracker_sha256=session.checkpoint_sha256,
                physics_runtime=runtime(),tracker_code=session.tracker_code,
                joystick=planner.provenance,tracker_checkpoint=session.checkpoint_identity),
            status='running', recording_complete=False, recording_body_hz=50,
            body_pose_convention='tracker_world XYZ + quaternion xyzw',
            measured_heading=measured_heading,
            measured_heading_policy=('actual_yaw_on_current_reference_sample' if measured_heading
                                     else 'reference_yaw_unmodified'),
            actuator_names=[model.actuator(i).name for i in range(model.nu)],
            actuator_joint_names=[model.joint(int(model.actuator_trnid[i,0])).name for i in range(model.nu)])
        if manifest['actuator_joint_names'] != list(session.names[0]):
            raise ValueError('PD actuator order differs from named joint targets')
        write_json(output/'manifest.json', manifest)
        write_json(output/'task.json', task)
        history = ProprioHistory(state, 0)
        tick = 0
        trajectory = PhysicalTrajectory([model.body(i).name for i in range(1,model.nbody)])
        def capture():
            trajectory.append(tick, session.physical_pose().cpu().numpy(),
                session.env.view('qpos')[0].cpu().numpy(), session.env.view('qvel')[0].cpu().numpy())
        capture()
        end_reason = 'timeout'
        execution_error = None
        all_clearance, goal_distances, warnings = [], [], []
        status = 'failed'
        goal_tolerance = float(task.get('goal_tolerance_m', .35))
        max_ticks = round(float(task.get('max_sim_seconds', task.get('max_seconds', 180)))*50)
        chunk_index = 0
        command = player.act(physical(), None, 0.)
        try:
            while tick < max_ticks:
                if cancellation.reason:
                    status, end_reason = 'aborted', cancellation.reason
                    break
                if replan_steps is not None and tick:
                    command = player.act(physical(), None, tick/50)
                facing = math.atan2(float(command.facing[0,1]), float(command.facing[0,0]))
                speed = max(0., float(command.speed[0]))
                dx, dy = (command.movement[0,:2]*speed).tolist()
                c, s = math.cos(facing), math.sin(facing)
                vx, vy = c*dx+s*dy, -s*dx+c*dy
                observation = history.snapshot()
                observation['joint_names'] = list(session.names[0])
                w, x, y, z = state[0,3:7].cpu().numpy()
                heading = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                observation['command_heading_delta'] = math.atan2(math.sin(facing-heading), math.cos(facing-heading))
                local_ref = yield ('plan', planner, vx, vy, observation)
                if int(local_ref.get('observation_tick', -1)) != tick or not local_ref.get('includes_current_anchor'):
                    raise ValueError('Reference not anchored at actual decision tick')
                order = [list(local_ref['joint_names']).index(name) for name in session.names[0]]
                local_ref = dict(local_ref, qpos=np.c_[local_ref['qpos'][:,:7], local_ref['qpos'][:,7:][:,order]],
                    joint_velocity=local_ref['joint_velocity'][:,order], joint_names=np.asarray(session.names[0]))
                ref = place_from_proprio(local_ref, state[0].cpu().numpy())
                session.queue = None
                session.append(ref)
                folder = output / f'chunk-{chunk_index:06d}'
                folder.mkdir()
                np.savez_compressed(folder/'reference_local.npz', **local_ref)
                np.savez_compressed(folder/'reference.npz', **ref)
                np.savez_compressed(folder/'input.npz', **observation)
                actual = [state[0,:size].cpu().numpy().copy()]
                actions, targets, masks = [], [], []
                start = tick
                planned = len(ref['qpos'])-1
                committed = min(planned, replan_steps) if replan_steps is not None else planned
                try:
                    for _ in range(min(committed, max_ticks-tick)):
                        if tick > 0 and tick % 50 == 0 and (replan_steps is None or tick != start):
                            command = player.act(physical(), None, tick/50)
                        if cancellation.reason:
                            status, end_reason = 'aborted', cancellation.reason
                            break
                        state, done = yield session
                        tick += 1
                        capture()
                        history.add(state, tick)
                        actual.append(state[0,:size].cpu().numpy().copy())
                        actions.append(session.last_action[0].cpu().numpy().copy())
                        targets.append(session.last_pd_target[0].cpu().numpy().copy())
                        masks.append(session.lookahead_valid.cpu().numpy().copy())
                        xy = physical().root_pos[0,:2].cpu().numpy()
                        cell = nav.cells(xy)
                        inside = 0 <= cell[0] < nav.shape[1] and 0 <= cell[1] < nav.shape[0]
                        distance = float(np.linalg.norm(xy-route[-1]))
                        all_clearance.append(float(nav.clearance[cell[1],cell[0]]) if inside else 0.)
                        goal_distances.append(distance)
                        warnings.append(bool(done.item()))
                        if distance <= goal_tolerance:
                            status, end_reason = 'complete', 'goal_reached'
                            break
                        if not inside or all_clearance[-1] < float(task.get('minimum_clearance_m', task.get('minimum_actual_clearance_m', .3))):
                            end_reason = 'outside_map' if not inside else 'clearance_violation'
                            break
                        if float(state[0,2]) < .35 or not torch.isfinite(state).all():
                            end_reason = 'fallen' if torch.isfinite(state).all() else 'nonfinite_state'
                            break
                finally:
                    np.savez_compressed(folder/'execution.npz', actual=np.asarray(actual), policy_action=np.asarray(actions),
                        pd_target=np.asarray(targets), lookahead_valid=np.asarray(masks), state_ticks=np.arange(start,tick+1),
                        state_physics_steps=np.arange(start,tick+1)*4, state_timestamps=np.arange(start,tick+1)/50)
                    write_json(folder/'manifest.json', dict(schema='omega.playground.executed-chunk/1', chunk_index=chunk_index,
                        start_tick=start, physics_start_step=start*4, timestamp=start/50, planned_control_steps=planned,
                        executed_control_steps=tick-start, end_tick=tick, end_timestamp=tick/50, physics_end_step=tick*4,
                        embodiment=robot, joint_names=list(session.names[0]), world_from_chunk=ref['world_from_chunk'].tolist(),
                        scene_from_tracker=scene_from_tracker.tolist(), status='complete' if tick-start==planned else 'closed',
                        command=dict(vx=vx,vy=vy,facing=facing), reference_hz=50, includes_current_anchor=True))
                    if not actions:
                        folder.rename(output / f'unexecuted-{chunk_index:06d}')
                write_json(output/'progress.json', dict(tick=tick, goal_distance_m=goal_distances[-1] if goal_distances else None, position=physical().root_pos[0,:2].tolist(), player=player.status()))
                print(json.dumps(dict(tick=tick, goal_distance_m=goal_distances[-1] if goal_distances else None, position=physical().root_pos[0,:2].tolist())), flush=True)
                chunk_index += 1
                if status in ('complete', 'aborted') or end_reason != 'timeout' or tick-start < committed:
                    break
        except Exception as exc:
            status, end_reason = 'failed', 'execution_error'
            execution_error = dict(type=type(exc).__name__, message=str(exc))
        trajectory.save(output)
        np.savez_compressed(output/'terminal_state.npz', state=state[0,:size].cpu().numpy(),
            physics_step=tick*4, timestamp=tick/50, body_q=session.physical_pose().cpu().numpy(),
            qpos=session.env.view('qpos')[0].cpu().numpy(), qvel=session.env.view('qvel')[0].cpu().numpy())
        np.savez_compressed(output/'execution_metrics.npz', root_clearance_m=all_clearance, goal_distance_m=goal_distances, tracking_warning=warnings)
        manifest.update(status=status, control_steps=tick, end_reason=end_reason, execution_error=execution_error)
        manifest['task']['result'] = 'success' if status == 'complete' else ('aborted' if status == 'aborted' else 'failure')
        manifest['task']['end_reason'] = end_reason
        manifest['task']['metrics'] = dict(goal_distance_m=goal_distances[-1] if goal_distances else None, min_root_clearance_m=min(all_clearance) if all_clearance else None, tracking_warning_count=sum(warnings))
        write_json(output/'manifest.json', manifest)
        return manifest
    finally:
        planner.close()


def run(task, output, cancellation=None):
    recording = record_task(task, output, cancellation)
    try:
        request = next(recording)
        while True:
            if isinstance(request, tuple):
                _, planner, vx, vy, observation = request
                response = planner.generate(vx, vy, observation=observation)
            else:
                response = request.step(measured_heading=task.get('measured_heading', True), streaming=True)
            request = recording.send(response)
    except StopIteration as finished:
        return finished.value
    finally:
        recording.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with Cancellation().installed() as cancellation:
        result = run(json.loads(args.task.read_text()), args.output, cancellation)
    print(json.dumps(result['task']))
    raise SystemExit(0 if result['status']=='complete' else 2)
