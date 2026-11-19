"""Recompute command metrics from saved references and fingerprint saved model evidence."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from joystick_reference import audit, dump, ROOT
from matrix import matrix
from omega_playground.policy.supermimic.provenance import file_identity, repository


def summarize(root):
    config=json.loads((ROOT/'configs/acceptance/supermimic.json').read_text())
    summary=[]
    for robot in config['embodiments']:
        folder=root/'layer2'/robot
        result=json.loads((folder/'result.json').read_text())
        provenance=json.loads((folder/'provenance.json').read_text())
        model=mujoco.MjModel.from_xml_path(provenance['xml']['path'])
        for row in result['cases']:
            case=folder/row['case']
            with np.load(case/'reference.npz',allow_pickle=False) as archive:ref={k:archive[k] for k in archive.files}
            with np.load(case/'input.npz',allow_pickle=False) as archive:obs={k:archive[k] for k in archive.files}
            obs['joint_names']=list(obs['joint_names'])
            command=json.loads((case/'command.json').read_text())
            row.update(audit(ref,obs,model,command,config['layer2']['quaternion_norm_atol']))
            dump(case/'metrics.json',{k:v for k,v in row.items() if k!='case'})
        result['metric_analysis_code']=repository(ROOT)
        dump(folder/'result.json',result)
        tracker=root/'layer1'/robot
        hashes={str(p.relative_to(root)):file_identity(p) for p in tracker.glob('*/physics.*')}
        hashes['reference50.npz']=file_identity(tracker/'reference50.npz')
        dump(tracker/'resource-hashes.json',hashes)
        with np.load(tracker/'reference50.npz') as ref:joints=len(ref['joint_names'])
        with np.load(tracker/'reference_comparison.npz') as ref:
            actor_parts_equal=bool(np.array_equal(ref['native'][:,:13+2*joints],ref['playground'][:,:13+2*joints]))
        with np.load(tracker/'native/trace.npz') as n,np.load(tracker/'playground/trace.npz') as p:
            analysis=dict(actor_reference_parts_equal=actor_parts_equal,
                basis='SMK Layout.goals consumes root, joint q and qdot; it discards appended FK body states',
                initial_history_equal=bool(np.array_equal(n['history'][0],p['history'][0])),
                initial_action_equal=bool(np.array_equal(n['action'][0],p['action'][0])),
                first_step_qvel_max_difference=float(np.abs(n['qvel'][1]-p['qvel'][1]).max()))
        dump(tracker/'input-analysis.json',analysis)
        tr=json.loads((tracker/'result.json').read_text())
        summary.append(dict(robot=robot,tracker_status=tr['status'],root_difference_m=tr['max_root_difference_m'],
            joystick_status=result['status'],joystick_cases=len(result['cases']),
            max_anchor_joint_velocity_jump_rad_s=max(r['anchor_joint_velocity_error_rad_s'] for r in result['cases']),
            max_estimated_foot_penetration_m=max(r['estimated_foot_penetration_m'] or 0 for r in result['cases'])))
    dump(root/'summary.json',summary)
    dump(root/'matrix.json',matrix(root,config['embodiments']))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    summarize(p.parse_args().root)
