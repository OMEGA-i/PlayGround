"""Matched-initial-state reference replays to isolate Joystick and Tracker defects."""
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .heading import HeadingFrame
from .live_client import JoystickClient
from .release import tracker_checkpoint
from .rollout import TrackingSession


def motion_reference(path, robot, frames=None):
    with np.load(path,allow_pickle=False) as archive:
        if float(archive['fps']) != 50:
            raise ValueError('Diagnostic source must be native 50 Hz motion')
        state, names = archive['state'].copy(), archive['joint_names'].copy()
    if frames is not None:
        state = state[frames]
    n = len(names)
    return dict(qpos=np.c_[state[:,:7],state[:,13:13+n]],
        root_linear_velocity=state[:,7:10],root_angular_velocity_world=state[:,10:13],
        joint_velocity=state[:,13+n:13+2*n],joint_names=names,fps=50,embodiment=robot)


def auto_replay(robot, checkpoint, xml, output, root, turn=False, steps=500):
    """Exercise the live command adapter and 1 Hz Player without a web server/render."""
    spec = importlib.util.spec_from_file_location('live_diagnostic', root/'tools/supermimic/serve_supermimic_live.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from omega_playground.control.tensor_web_player import TensorWebPlayer
    from omega_playground.control.drawn_path import drawn_path
    world = module.World.__new__(module.World)
    world.tracker = TrackingSession(robot, checkpoint, xml)
    world.heading = HeadingFrame([0,0])
    world.view = SimpleNamespace(offset=torch.zeros(3,device='cuda:0'))
    world.mode, world.vx, world.vy, world.facing = 'auto', 0., 0., 0.
    world.tick = world.next_player = 0
    client = JoystickClient(robot, xml)
    output.mkdir(parents=True,exist_ok=False)
    try:
        world.tracker.append(world.heading.place(client.generate(0,0),0.))
        world.state = world.tracker.env.state()
        start = world.physical().root_pos[0,:2].tolist()
        plan = drawn_path([start,[start[0]+(0 if turn else 8),start[1]+(8 if turn else 0)]])
        config = json.loads((root/'configs/scenes/obstacle.json').read_text())
        config = {**config,**plan,'task_mode':'drawn_path','reference':{**config['reference'],'preserve_path':True}}
        world.player = TensorWebPlayer(config,'cuda:0')
        world.player.reset(torch.tensor([0],device='cuda:0'),world.physical())
        world.player.configure_locomotion('walk',.65,0.)
        states, poses, targets, commands = [], [], [], []
        for tick in range(steps):
            world.tick = tick
            world.select_command()
            if tick%50 == 0:
                commands.append(dict(time=tick/50,vx=world.vx,vy=world.vy,facing=world.facing,
                                     status=world.player.status(),actual=world.state[0,:13].tolist()))
            if len(world.tracker.queue)<=23:
                world.tracker.append(world.heading.place(client.generate(world.vx,world.vy),world.facing))
            targets.append(world.tracker.queue[:1].clone())
            world.state,_ = world.tracker.step(measured_heading=True)
            states.append(world.state.clone())
            poses.append(world.tracker.physical_pose().clone())
        np.savez_compressed(output/'rollout.npz',actual=torch.cat(states).cpu().numpy(),
                            reference=torch.cat(targets).cpu().numpy(),fps=50)
        np.savez_compressed(output/'body_states.npz',body_q=torch.stack(poses).cpu().numpy(),
                            sim_time=(np.arange(len(poses))+1)/50,
                            body_names=[world.tracker.env.cpu_model.body(i).name for i in range(1,world.tracker.env.cpu_model.nbody)])
        (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        print(json.dumps(dict(robot=robot,auto_commands=commands)),flush=True)
    finally:
        client.close()


def replay(robot, checkpoint, xml, ref, output, measured_heading, steps=500):
    output.mkdir(parents=True, exist_ok=False)
    session = TrackingSession(robot, checkpoint, xml)
    clip = session.append(ref)
    actual, poses, failures = [session.env.state().clone()], [session.physical_pose().clone()], []
    for _ in range(min(steps, len(clip)-11)):
        state, failed = session.step(measured_heading=measured_heading)
        actual.append(state.clone())
        poses.append(session.physical_pose().clone())
        failures.append(bool(failed.item()))
        if not torch.isfinite(state).all():
            break
    states = torch.cat(actual).cpu().numpy()
    target = clip[:len(states)].cpu().numpy()
    np.savez_compressed(output/'rollout.npz', actual=states, reference=target, fps=50)
    np.savez_compressed(output/'body_states.npz', body_q=torch.stack(poses).cpu().numpy(),
                        sim_time=np.arange(len(poses))/50,
                        body_names=[session.env.cpu_model.body(i).name for i in range(1,session.env.cpu_model.nbody)])
    error = np.linalg.norm(states[:,:3]-target[:,:3],axis=1)
    result = dict(robot=robot, measured_heading=measured_heading, steps=len(failures),
                  first_failure_s=(failures.index(True)+1)/50 if any(failures) else None,
                  failed_frames=sum(failures), root_rmse_m=float(np.sqrt(np.mean(error**2))),
                  actual_end_xyz=states[-1,:3].tolist(), reference_end_xyz=target[-1,:3].tolist(),
                  actual_displacement=(states[-1,:3]-states[0,:3]).tolist(),
                  reference_displacement=(target[-1,:3]-target[0,:3]).tolist(),
                  min_height_m=float(states[:,2].min()))
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


def diagnose(robot, output, shared, root, scenarios=None, steps=500):
    """Save once, replay twice. No renderer, scene contacts, or live-world mutation."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    xml = (shared/'joystick/assets/sih2300/robot.xml' if robot=='sih2300' else
           shared/'supermimic/assets/robots'/robot/'robot.xml')
    checkpoint = tracker_checkpoint(robot, shared)
    if scenarios in (['auto'], ['auto_turn']):
        auto_replay(robot, checkpoint, xml, output/'auto', root, turn=scenarios==['auto_turn'], steps=steps)
        return
    reports = []
    scenarios = scenarios or ('recorded', 'forward', 'idle_forward', 'idle')
    references = {}
    if 'recorded' in scenarios:
        source = shared/'datasets/lafan/motions'/robot/'walk3_subject3/motion.npz'
        references['recorded'] = motion_reference(source, robot, slice(1000,1511))
    bank = json.loads((root.parent/'Joystick/configs/soma_primitives_25fps.json').read_text())['primitives']
    for scenario in scenarios:
        if not scenario.startswith('source_'):
            continue
        primitive = {'source_forward':'walk_forward_medium','source_left':'walk_forward_left_medium','source_idle':'idle'}[scenario]
        motion = shared/'datasets/bones-seed/motions'/robot/Path(bank[primitive]['source_id']).with_suffix('')/'motion.npz'
        references[scenario] = motion_reference(motion, robot)
    generated = [s for s in scenarios if s!='recorded' and not s.startswith('source_')]
    client = JoystickClient(robot,xml) if generated else None
    try:
        for scenario in generated:
            frame = HeadingFrame([0,0])
            chunks = []
            for index in range((steps+11)//40+2):
                speed = 0. if scenario=='idle' or (scenario=='idle_forward' and index==0) else .65
                chunks.append(frame.place(client.generate(speed,0,reset=index==0),0.))
            arrays = ('qpos','root_linear_velocity','root_angular_velocity_world','joint_velocity')
            ref = {**chunks[0], **{key:np.concatenate([chunk[key] for chunk in chunks]) for key in arrays}}
            ref['timestamps'] = np.arange(len(ref['qpos']))/50
            ref['projection_error_p95_rad'] = np.array([float(c['projection_error_p95_rad']) for c in chunks])
            ref['clamped_joint_samples'] = np.array([int(c['clamped_joint_samples']) for c in chunks])
            references[scenario] = ref
    finally:
        if client:
            client.close()
    for name,ref in references.items():
        np.savez_compressed(output/f'{name}.npz',**ref)
        for anchor in (False,True):
            reports.append(dict(scenario=name, **replay(robot,checkpoint,xml,ref,
                output/f'{name}-{"live" if anchor else "native"}',anchor,steps)))
    (output/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
