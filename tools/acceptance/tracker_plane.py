"""Compare native SMK evaluator operations with the actual Playground adapter on native plane.

This is an instrumented evaluator-loop audit, not a substituted success metric.
No scene_xml, heading rewrite, collision override, parameter tuning or CPU simulation.
"""
import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import torch

from omega_playground.policy.supermimic.rollout import TrackingSession
from omega_playground.policy.supermimic.release import tracker_checkpoint
from omega_playground.policy.supermimic.provenance import file_identity, repository, runtime

ROOT = Path(__file__).resolve().parents[2]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return dict(equal=False, shape_a=list(a.shape), shape_b=list(b.shape))
    delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
    bad = np.argwhere(a != b)
    return dict(equal=bool(np.array_equal(a, b)), max_abs=float(delta.max(initial=0)),
        rms=float(np.sqrt(np.mean(delta**2))), first_index=bad[0].tolist() if len(bad) else None)


def cpu(t):
    return t.detach().cpu().numpy().copy()


class Trace:
    def __init__(self, env, output):
        self.env, self.output, self.rows = env, output, {}
        output.mkdir(parents=True)
        mujoco.mj_saveModel(env.cpu_model, str(output/'physics.mjb'), None)
        mujoco.mj_saveLastXML(str(output/'physics.xml'), env.cpu_model)
        self.contacts = []

    def add(self, key, value):
        self.rows.setdefault(key, []).append(cpu(value))

    def capture(self):
        torch.cuda.set_stream(self.env.torch_stream)
        for key in ('qpos', 'qvel', 'ctrl'):
            self.add(key, self.env.view(key))
        self.add('actual', self.env.state())
        self.add('bodies', self.env.body_states())
        # Actual GPU contacts at control boundaries; no CPU dynamics/reconstruction.
        data = self.env.data
        contact = data.contact
        entry = {}
        for name in ('ncon', 'nacon'):
            if hasattr(data, name):
                entry[name] = cpu(self.env.wp.to_torch(getattr(data, name)))
        for name in ('geom', 'dist', 'pos', 'worldid'):
            if hasattr(contact, name):
                entry[name] = cpu(self.env.wp.to_torch(getattr(contact, name)))
        self.contacts.append(entry)

    def save(self):
        arrays = {k:np.asarray(v) for k,v in self.rows.items()}
        np.savez_compressed(self.output/'trace.npz', **arrays)
        # Ragged contacts kept without pickle; one numeric archive per control boundary.
        folder = self.output/'contacts'; folder.mkdir()
        for i, contact in enumerate(self.contacts):
            np.savez_compressed(folder/f'{i:06d}.npz', **contact)
        bodies = arrays['bodies'][:, 0, 1:]
        np.savez_compressed(self.output/'body_states.npz',
            body_q=np.concatenate((bodies[:,:,:3], bodies[:,:,[4,5,6,3]]), axis=-1),
            body_names=[self.env.cpu_model.body(i).name for i in range(1,self.env.cpu_model.nbody)],
            sim_time=np.arange(len(bodies))/50)
        return arrays


@torch.inference_mode()
def run(robot, output, shared, config, native_control=False, deterministic=False,
        warp_deterministic=False):
    import warp as wp
    if warp_deterministic:
        # Set before importing modules which define Warp kernels.
        wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
    from supermimic.core import Actor, Layout
    from supermimic.data import Motions
    from supermimic.evaluation_schema import prepare_evaluation_motions
    from supermimic.mjwarp import MJWarp
    import supermimic.evaluate as evaluator
    import mujoco_warp as mw
    if deterministic:
        if not hasattr(mw,'DeterminismType'):
            raise RuntimeError('This MJWarp runtime does not support model-level determinism')
        # Set before MJWarp constructor warmup/CUDA graph capture, for BOTH worlds.
        # Use the official per-model controls, not blanket Warp kernel rewriting.
        original_put_model=mw.put_model
        def deterministic_model(*args,**kwargs):
            model=original_put_model(*args,**kwargs)
            model.opt.deterministic=mw.DeterminismType.ALL
            return model
        mw.put_model=deterministic_model
    settings = config['layer1']
    os.environ.pop('OMEGA_SUPERMIMIC_CONTACTS', None)
    torch.manual_seed(7)
    checkpoint = tracker_checkpoint(robot, shared)
    xml = shared/('joystick/assets/sih2300/robot.xml' if robot == 'sih2300'
                  else f'supermimic/assets/robots/{robot}/robot.xml')
    session = TrackingSession(robot, checkpoint, xml)
    ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    motion = shared/f'datasets/lafan/motions/{robot}/{settings["motion"]}/motion.npz'
    motions = Motions([motion], session.device)
    prepare_evaluation_motions(ck, motions, session.device)
    assert motions.names == session.names
    start, steps = settings['start_frame'], settings['control_steps']
    native_clip = motions.clips[0][start:start+steps+11]
    if len(native_clip) != steps+11:
        raise ValueError('Motion does not cover the complete audit window')
    size = len(session.names[0])
    a = cpu(native_clip)
    ref = dict(qpos=np.c_[a[:,:7],a[:,13:13+size]], root_linear_velocity=a[:,7:10],
        root_angular_velocity_world=a[:,10:13], joint_velocity=a[:,13+size:13+2*size],
        joint_names=session.names[0], embodiment=robot, fps=50)
    pg_clip = session.append(ref)
    native = MJWarp(ck['config'], motions.names, 1, session.device)
    native.layout = Layout(native.layout.joints, native.layout.bodies)
    actor = Actor(native.layout, ck['args']['small']).to(session.device).eval()
    actor.load_state_dict(ck['actor'], strict=True)
    native.reset(torch.zeros(1,device=session.device,dtype=torch.long), native_clip[:1])
    prev = torch.zeros(1,size,device=session.device)
    history = native.layout.obs(native.state(),prev,native.default,False,goal=native_clip[:1])[:,None].repeat(1,10,1)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'reference50.npz', **ref)
    np.savez_compressed(output/'reference_comparison.npz', native=cpu(native_clip), playground=cpu(pg_clip))
    native_trace, pg_trace = Trace(native,output/'native'), Trace(session.env,output/'playground')
    models = []
    for env in (native,session.env):
        m = env.cpu_model
        gid = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'supermimic_ground')
        if gid < 0 or m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError('Native analytic plane required')
        models.append(dict(plane_geom_id=gid, plane_type=int(m.geom_type[gid]),
            deterministic=int(getattr(env.model.opt,'deterministic',0)),
            timestep=m.opt.timestep,disableflags=int(m.opt.disableflags),solver=int(m.opt.solver),
            kp=m.actuator_gainprm[:,0].tolist(),kd=(-m.actuator_biasprm[:,2]).tolist(),
            friction=m.geom_friction.tolist(), action_scale=cpu(env.scale).tolist()))
    dump(output/'provenance.json',dict(robot=robot, checkpoint=session.checkpoint_identity,
        xml=file_identity(xml), motion=file_identity(motion), evaluator=file_identity(evaluator.__file__),
        playground=repository(ROOT), tracker=session.tracker_code, runtime=runtime(),
        config=config, checkpoint_physics_config=ck['config'], models=models,
        contact_sampling='actual GPU contact arrays at 50 Hz control boundaries; not all 200 Hz substeps',
        measured_heading=False, scene_xml=None, native_control=native_control,
        deterministic_requested=deterministic,
        warp_deterministic=str(wp.config.deterministic)))
    if deterministic and any(m['deterministic']!=int(mw.DeterminismType.ALL) for m in models):
        raise RuntimeError('Deterministic mode was not applied to both simulation worlds')
    def capture_actor(_module, args):
        pg_trace.add('history',args[0]); pg_trace.add('window',args[1])
    hook = session.actor.register_forward_pre_hook(capture_actor)
    native_trace.capture(); pg_trace.capture()
    try:
        for t in range(steps):
            # MJWarp owns a dedicated Torch/Warp stream. Two simulator instances
            # must each execute their actor, control write and physics on that stream.
            torch.cuda.set_stream(native.torch_stream)
            # Exactly the evaluator Motions.get window, including its source FK.
            goals = motions.get(torch.zeros(1,device=session.device,dtype=torch.long),
                torch.tensor([start+t],device=session.device),11)
            native_trace.add('history',history); native_trace.add('window',goals)
            action = actor(history,goals)
            native_trace.add('action',action)
            state = native.step(action)
            target = native_clip[t+1:t+2]
            native_trace.add('done',native.layout.done(state,target))
            history = torch.cat((history[:,1:],native.layout.obs(state,action,native.default,False,goal=target)[:,None]),1)
            torch.cuda.set_stream(session.env.torch_stream)
            if native_control:
                # Repeat the evaluator operations in the second native world.
                # This measures a native/native numerical baseline, not adapter acceptance.
                pg_action=session.actor(session.history,goals)
                pg_state=session.env.step(pg_action)
                done=session.env.layout.done(pg_state,target)
                session.history=torch.cat((session.history[:,1:],session.env.layout.obs(
                    pg_state,pg_action,session.env.default,False,goal=target)[:,None]),1)
                session.last_action=pg_action.clone()
            else:
                _, done = session.step(measured_heading=False)
            pg_trace.add('action',session.last_action); pg_trace.add('done',done)
            native_trace.capture(); pg_trace.capture()
            if t % 50 == 0:
                print(f'{robot} layer1 {t+1}/{steps}',flush=True)
    finally:
        hook.remove()
        x,y = native_trace.save(),pg_trace.save()
    comparisons = {k:compare(x[k],y[k]) for k in x}
    comparisons['reference'] = compare(cpu(native_clip),cpu(pg_clip))
    archive_equal = all(v['equal'] for v in comparisons.values()) and models[0] == models[1]
    actor_reference_equal=compare(cpu(native_clip[:,:13+2*size]),cpu(pg_clip[:,:13+2*size]))
    execution_equal=all(v['equal'] for k,v in comparisons.items() if k not in ('reference','window'))
    passed=execution_equal and actor_reference_equal['equal'] and models[0]==models[1]
    status=('passed' if passed else 'failed') if deterministic else 'not_applicable_nondeterministic'
    result = dict(robot=robot,layer=1,status=status,native_control=native_control,
        deterministic_requested=deterministic,actor_reference_comparison=actor_reference_equal,
        execution_bitwise_equal=execution_equal,complete_archive_bitwise_equal=archive_equal,
        acceptance_basis='Deterministic mode required; actor-consumed reference plus actual execution. Derived FK-only tails reported separately.',
        comparisons=comparisons,model_parameters_equal=models[0]==models[1],steps=steps,
        native_done_count=int(x['done'].sum()),playground_done_count=int(y['done'].sum()),
        max_root_difference_m=float(np.linalg.norm(x['actual'][:,:,:3]-y['actual'][:,:,:3],axis=-1).max()),
        meaning='Adapter equivalence only; does not certify locomotion or contact safety')
    dump(output/'result.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shared',type=Path,default=Path.home()/'omega-assets')
    p.add_argument('--config',type=Path,default=ROOT/'configs/acceptance/supermimic.json')
    p.add_argument('--native-control',action='store_true')
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--warp-deterministic',action='store_true',
        help='Also enable Warp RUN_TO_RUN before loading simulator kernels')
    p.add_argument('--steps',type=int,default=None)
    args=p.parse_args()
    config=json.loads(args.config.read_text())
    if args.steps is not None:
        if args.steps<1:p.error('--steps must be positive')
        config['layer1']['control_steps']=args.steps
    run(args.robot,args.output,args.shared,config,args.native_control,args.deterministic,
        args.warp_deterministic)
