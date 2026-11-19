"""Gated native-plane Joystick/Tracker command suite, with actual-state replanning."""
import argparse
import json
import math
from pathlib import Path
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from tracker_plane import ROOT, Trace, dump
from omega_playground.policy.supermimic.rollout import TrackingSession
from omega_playground.policy.supermimic.release import tracker_checkpoint
from omega_playground.policy.supermimic.live_client import JoystickClient
from omega_playground.policy.supermimic.chunk import ProprioHistory
from omega_playground.policy.supermimic.heading import place_from_proprio


@torch.inference_mode()
def run(robot,root,shared,diagnostic=False):
    output=root/'layer3'/robot
    output.mkdir(parents=True,exist_ok=False)
    prerequisites={}
    for layer in (1,2):
        p=root/f'layer{layer}'/robot/'result.json'
        prerequisites[str(layer)]=json.loads(p.read_text())['status'] if p.exists() else 'not_run'
    if not diagnostic and any(v!='passed' for v in prerequisites.values()):
        dump(output/'result.json',dict(robot=robot,layer=3,status='blocked_prerequisites',prerequisites=prerequisites))
        return
    cfg=json.loads((ROOT/'configs/acceptance/supermimic.json').read_text())['layer3']
    xml=shared/('joystick/assets/sih2300/robot.xml' if robot=='sih2300' else f'supermimic/assets/robots/{robot}/robot.xml')
    os.environ.pop('OMEGA_SUPERMIMIC_CONTACTS',None)
    session=TrackingSession(robot,tracker_checkpoint(robot,shared),xml)
    planner=JoystickClient(robot,xml)
    with np.load(root/'layer1'/robot/'reference50.npz') as archive:base={k:archive[k].copy() for k in archive.files}
    summaries=[]
    try:
        for seed in cfg['seeds']:
            for yaw in cfg['initial_yaws_rad']:
                for commit in cfg['replan_control_steps']:
                    case=output/f'seed{seed}-yaw{yaw:.3f}-commit{commit}'
                    trace=Trace(session.env,case)
                    ref={k:v.copy() for k,v in base.items()}
                    rotation=Rotation.from_euler('z',yaw)
                    ref['qpos'][:,:3]=rotation.apply(ref['qpos'][:,:3])
                    ref['qpos'][:,3:7]=(rotation*Rotation.from_quat(ref['qpos'][:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
                    for key in ('root_linear_velocity','root_angular_velocity_world'):ref[key]=rotation.apply(ref[key])
                    session.history=session.queue=None
                    session.append(ref)
                    state=session.env.state();history=ProprioHistory(state,0)
                    tick=0;failed=False;chunks=[]
                    trace.capture()
                    def hook(_m,inputs):
                        trace.add('history',inputs[0]);trace.add('window',inputs[1])
                    handle=session.actor.register_forward_pre_hook(hook)
                    try:
                        for command in cfg['commands']:
                            remaining=round(command['seconds']*50)
                            w,x,y,z=state[0,3:7].cpu().numpy()
                            desired=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))+command['heading_delta']
                            while remaining>0 and not failed:
                                obs=history.snapshot();obs['joint_names']=session.names[0]
                                w,x,y,z=state[0,3:7].cpu().numpy()
                                actual_yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                                obs['command_heading_delta']=math.atan2(math.sin(desired-actual_yaw),math.cos(desired-actual_yaw))
                                local=planner.generate_many([dict(env_id=0,reset=tick==0,seed=seed,
                                    velocity=[command['vx'],command['vy']],observation=obs)])[0]
                                reference=place_from_proprio(local,state[0].cpu().numpy())
                                chunk=case/f'chunk-{len(chunks):04d}';chunk.mkdir()
                                np.savez_compressed(chunk/'input.npz',**obs)
                                np.savez_compressed(chunk/'reference_local.npz',**local)
                                np.savez_compressed(chunk/'reference_world.npz',**reference)
                                dump(chunk/'command.json',command)
                                session.queue=None;session.append(reference)
                                count=min(commit,remaining,len(reference['qpos'])-1)
                                chunks.append(dict(tick=tick,count=count,command=command))
                                for _ in range(count):
                                    state,done=session.step(measured_heading=False,streaming=True)
                                    trace.add('action',session.last_action);trace.add('done',done)
                                    tick+=1;remaining-=1;history.add(state,tick);trace.capture()
                                    failed=bool(done.item())
                                    if failed:break
                            if failed:break
                    finally:
                        handle.remove();trace.save();dump(case/'chunks.json',chunks)
                    summaries.append(dict(case=case.name,steps=tick,failed=failed))
    finally:planner.close()
    dump(output/'result.json',dict(robot=robot,layer=3,status='diagnostic_only' if diagnostic else 'requires_contact_and_response_review',
        prerequisites=prerequisites,cases=summaries,joystick=planner.provenance,tracker=session.checkpoint_identity,
        tracker_code=session.tracker_code,scene='native_analytic_plane',measured_heading=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--robot',required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--shared',type=Path,default=Path.home()/'omega-assets')
    p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args();run(a.robot,a.root,a.shared,a.diagnostic)
