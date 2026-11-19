"""Independent reference-generation audit; never uses Tracker success as a proxy."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from omega_playground.policy.supermimic.live_client import JoystickClient
from omega_playground.policy.supermimic.heading import place_from_proprio
from omega_playground.policy.supermimic.provenance import file_identity, repository

ROOT=Path(__file__).resolve().parents[2]


def dump(p, x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def kinematics(reference, model, output):
    """FK-only estimates; no dynamics or claim of measured physical contact."""
    data=mujoco.MjData(model)
    names=list(reference['joint_names'])
    slots=[model.jnt_qposadr[model.joint(n).id] for n in names]
    feet=[i for i in range(model.nbody) if any(token in model.body(i).name.lower()
           for token in ('foot','ankle','toe'))]
    geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g] in feet]
    bodies=[];floor=[]
    for q in reference['qpos']:
        data.qpos[:7]=q[:7];data.qpos[slots]=q[7:]
        mujoco.mj_kinematics(model,data)
        bodies.append(np.c_[data.xpos[1:],data.xquat[1:,[1,2,3,0]]].copy())
        lower=[]
        for g in geoms:
            pos=data.geom_xpos[g];rot=data.geom_xmat[g].reshape(3,3);size=model.geom_size[g]
            kind=model.geom_type[g]
            if kind==mujoco.mjtGeom.mjGEOM_SPHERE:extent=size[0]
            elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=size[0]+size[1]*abs(rot[2,2])
            elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:extent=size[0]*np.sqrt(max(0,1-rot[2,2]**2))+size[1]*abs(rot[2,2])
            elif kind==mujoco.mjtGeom.mjGEOM_BOX:extent=np.abs(rot[2])@size
            elif kind==mujoco.mjtGeom.mjGEOM_MESH:
                mesh=model.geom_dataid[g];start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
                lower.append(float(pos[2]+(model.mesh_vert[start:start+count]@rot[2]).min()));continue
            else:continue
            lower.append(float(pos[2]-extent))
        floor.append(min(lower) if lower else np.nan)
    bodies=np.asarray(bodies);floor=np.asarray(floor)
    speed=np.linalg.norm(np.diff(bodies[:,np.asarray(feet)-1,:2],axis=0)*50,axis=-1)
    np.savez_compressed(output/'kinematics.npz',body_q=bodies,body_names=[model.body(i).name for i in range(1,model.nbody)],
        minimum_foot_geometry_z=floor,foot_body_speed_xy=speed,sim_time=np.arange(len(bodies))/50)
    # The 2 cm proximity band is only an explicitly labelled slip proxy, not a gate.
    near=(floor[:-1]<=.02)&(floor[1:]<=.02)
    return dict(foot_geometry_min_z_m=float(np.nanmin(floor)) if len(geoms) else None,
        estimated_foot_penetration_m=float(max(0,-np.nanmin(floor))) if len(geoms) else None,
        foot_speed_near_floor_proxy_m_s=float(speed[near].mean()) if near.any() and speed.size else None,
        foot_proxy_band_m=.02,foot_estimate='FK geometry only; not physical contact',foot_bodies=[model.body(i).name for i in feet])


def audit(ref, observation, model, command, tolerance):
    q=np.asarray(ref['qpos']); v=np.asarray(ref['joint_velocity'])
    names=list(ref['joint_names']); ids=[model.joint(n).id for n in names]
    ranges=model.jnt_range[ids]; limited=model.jnt_limited[ids].astype(bool)
    violation=np.maximum(ranges[:,0]-q[:,7:],q[:,7:]-ranges[:,1])[:,limited]
    speed=np.linalg.norm(np.diff(q[:,:2],axis=0)*50,axis=-1)
    target_speed=float(np.hypot(command['vx'],command['vy']))
    w,x,y,z=q[:,3:7].T
    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    expected=command['heading_delta']
    error=np.arctan2(np.sin(yaw-yaw[0]-expected),np.cos(yaw-yaw[0]-expected))
    heading=yaw[0]+expected
    desired=np.array([np.cos(heading)*command['vx']-np.sin(heading)*command['vy'],
        np.sin(heading)*command['vx']+np.cos(heading)*command['vy']])
    vector_error=np.diff(q[:,:2],axis=0)*50-desired
    heading_hits=np.flatnonzero(np.abs(error)<=abs(expected)*.5) if abs(expected)>1e-8 else []
    initial_speed=float(np.linalg.norm(observation['states'][-1,7:9]))
    speed_hits=np.flatnonzero(np.abs(speed-target_speed)<=abs(initial_speed-target_speed)*.5) if abs(initial_speed-target_speed)>1e-8 else []
    actual=observation['states'][-1]
    ordered=[observation['joint_names'].index(n) for n in names]
    finite=all(np.isfinite(ref[k]).all() for k in ('qpos','joint_velocity','root_linear_velocity','root_angular_velocity_world'))
    quat_error=float(np.max(np.abs(np.linalg.norm(q[:,3:7],axis=-1)-1)))
    limits=max(0.,float(violation.max(initial=0)))
    interface=(int(ref['fps'])==50 and bool(ref.get('includes_current_anchor'))
        and int(ref.get('observation_tick',-1))==int(observation['ticks'][-1])
        and len(names)==len(set(names)) and set(names)==set(observation['joint_names'])
        and len(q)==int(ref['planner_future_frames'])*2+1
        and np.allclose(np.diff(ref['timestamps']),.02,rtol=0,atol=1e-12))
    return dict(interface=interface,finite=finite,quaternion_norm_max_error=quat_error,
        quaternion_valid=quat_error<=tolerance,joint_limit_max_violation_rad=limits,
        joint_limits_valid=limits==0.,speed_rmse_m_s=float(np.sqrt(np.mean((speed-target_speed)**2))),
        generated_mean_speed_m_s=float(speed.mean()),command_speed_m_s=target_speed,
        velocity_xy_rmse_m_s=np.sqrt(np.mean(vector_error**2,axis=0)).tolist(),
        heading_half_response_s=float(heading_hits[0]/50) if len(heading_hits) else None,
        speed_half_response_s=float(speed_hits[0]/50) if len(speed_hits) else None,
        response_latency_definition='first sample within half of requested heading/speed change; diagnostic, no pass threshold',
        heading_final_error_rad=float(error[-1]),root_displacement_m=float(np.linalg.norm(q[-1,:3]-q[0,:3])),
        root_accel_max_m_s2=float(np.linalg.norm(np.diff(q[:,:3],n=2,axis=0)*2500,axis=-1).max()),
        joint_velocity_max_rad_s=float(np.abs(v).max()),joint_acceleration_max_rad_s2=float(np.abs(np.diff(v,axis=0)*50).max()),
        anchor_joint_error_rad=float(np.abs(q[0,7:]-actual[13+np.asarray(ordered)]).max()),
        anchor_joint_velocity_error_rad_s=float(np.abs(v[0]-actual[13+len(names)+np.asarray(ordered)]).max()),
        root_velocity_consistency_max_m_s=float(np.abs(np.diff(q[:,:3],axis=0)*50-ref['root_linear_velocity'][:-1]).max()))


def run(robot, output, shared, config):
    settings=config['layer2']
    xml=shared/('joystick/assets/sih2300/robot.xml' if robot=='sih2300' else f'supermimic/assets/robots/{robot}/robot.xml')
    motion=shared/f'datasets/lafan/motions/{robot}/walk3_subject3/motion.npz'
    model=mujoco.MjModel.from_xml_path(str(xml))
    with np.load(motion,allow_pickle=False) as data:
        states=data['state'].copy(); names=data['joint_names'].tolist()
    output.mkdir(parents=True,exist_ok=False)
    client=JoystickClient(robot,xml)
    dump(output/'provenance.json',dict(joystick=client.provenance,xml=file_identity(xml),
        motion=file_identity(motion),playground=repository(ROOT),config=config,
        observation_source='offline motion initial history; subsequent history is generated reference, never physical rollout',
        units='metres, seconds, radians; root world XYZ, quaternion wxyz; 50 Hz output'))
    results=[]
    try:
        for start in settings['motion_start_frames']:
            for seed in settings['seeds']:
                history=states[np.arange(start-6,start+1)].copy()
                tick=0
                for ci,command in enumerate(settings['commands']):
                    case=output/f'frame{start}-seed{seed}-{ci:02d}-{command["name"]}';case.mkdir()
                    obs=dict(states=history[::2],ticks=np.arange(tick-6,tick+1,2),
                        valid=np.ones(4,dtype=bool),physics_steps=np.arange(tick-6,tick+1,2)*4,
                        timestamp=np.arange(tick-6,tick+1,2)/50,joint_names=names,
                        command_heading_delta=command['heading_delta'])
                    request=dict(env_id=0,seed=seed,reset=ci==0,velocity=[command['vx'],command['vy']],observation=obs)
                    ref=client.generate_many([request])[0]
                    np.savez_compressed(case/'input.npz',**obs)
                    np.savez_compressed(case/'reference.npz',**ref)
                    dump(case/'command.json',command)
                    metrics=audit(ref,obs,model,command,settings['quaternion_norm_atol'])
                    metrics.update(kinematics(ref,model,case))
                    metrics['projection_error_p95_rad']=float(ref['projection_error_p95_rad'])
                    metrics['clamped_joint_samples']=int(ref['clamped_joint_samples'])
                    if ci==0:
                        repeated=client.generate_many([request])[0]
                        metrics['same_seed_reference_bitwise_equal']=all(np.array_equal(ref[k],repeated[k]) for k in
                            ('qpos','joint_velocity','root_linear_velocity','root_angular_velocity_world'))
                        np.savez_compressed(case/'repeat.npz',**repeated)
                    placed=place_from_proprio(ref,history[-1])
                    np.savez_compressed(case/'reference_world.npz',**placed)
                    metrics['placed_root_anchor_error_m']=float(np.linalg.norm(placed['qpos'][0,:3]-history[-1,:3]))
                    dump(case/'metrics.json',metrics)
                    results.append(dict(case=case.name,**metrics))
                    # Execute no physics: synthetically advance one second of the reference.
                    order=[list(placed['joint_names']).index(n) for n in names]
                    q=placed['qpos']; v=placed['joint_velocity']
                    history=np.c_[q[44:51,:7],placed['root_linear_velocity'][44:51],
                        placed['root_angular_velocity_world'][44:51],q[44:51,7:][:,order],v[44:51][:,order]]
                    tick+=50
                    print(f'{robot} joystick {case.name}: {metrics}',flush=True)
    finally:
        client.close()
    failures=[r['case'] for r in results if not all(r[k] for k in ('interface','finite','quaternion_valid','joint_limits_valid'))
              or r.get('same_seed_reference_bitwise_equal') is False]
    result=dict(robot=robot,layer=2,status='failed' if failures else 'blocked_threshold_contract',
        failures=failures,cases=results,unvalidated=['command_response_thresholds','foot_sliding_and_clearance_thresholds',
        'joint_acceleration_thresholds','real_rollout_history'],physical_execution=False)
    dump(output/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shared',type=Path,default=Path.home()/'omega-assets')
    p.add_argument('--config',type=Path,default=ROOT/'configs/acceptance/supermimic.json')
    args=p.parse_args();run(args.robot,args.output,args.shared,json.loads(args.config.read_text()))
