"""Audit the private web contract and replay held-out walking through the live Tracker."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from omega_playground.policy.supermimic.release import tracker_checkpoint, tracker_release
from omega_playground.policy.supermimic.rollout import run


def verify(robot, contracts, output, shared):
    from supermimic.core import Actor, Layout, validate_checkpoint
    from supermimic.mjwarp import build_model
    from supermimic.physx import actuator_gains

    entry = tracker_release()[robot]
    checkpoint = tracker_checkpoint(robot, shared)
    ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    validate_checkpoint(ck)
    with checkpoint.open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == entry["sha256"]
    assert ck["iteration"] + 1 == entry["completed_iterations"]
    assert ck["training_contract"]["config"] == ck["config"]
    cfg = ck['config']
    # Native deployments load the frozen Actor directly, without publishing web weights.
    native = contracts is None
    if native:
        layout = Layout(len(ck['joint_names']), len(ck['body_names']))
        actor = Actor(layout, ck['args']['small']).eval()
        actor.load_state_dict(ck['actor'], strict=True)
        for value in actor.state_dict().values():
            if value.is_floating_point():
                assert torch.isfinite(value).all()
        meta = dict(source_asset_sha256=ck['training_contract']['asset_sha256'],
            checkpoint_source=str(checkpoint), joint_names=ck['joint_names'],body_names=ck['body_names'],
            physics_dt=.005,control_dt=.02,substeps=4,history_length=10,goal_frames=11,
            layout=dict(obs_dim=layout.obs_dim,dim=layout.dim),
            normalizers={k:dict(mean=getattr(actor,k+'_norm').mean.tolist(),
                               std=getattr(actor,k+'_norm').std().tolist()) for k in ('obs','goal')})
    else:
        meta = contracts[robot]
    xml = shared / ('joystick/assets/sih2300/robot.xml' if robot == 'sih2300'
                    else f'supermimic/assets/robots/{robot}/robot.xml')
    assert hashlib.sha256(xml.read_bytes()).hexdigest() == meta['source_asset_sha256'] == entry['training_xml_sha256']
    assert Path(meta['checkpoint_source'].split(':')[0]).as_posix().endswith('/' + entry['relative_path'])
    assert ck['joint_names'] == meta['joint_names'] == entry['joint_names']
    assert ck['body_names'] == meta['body_names'] == entry['body_names']
    assert (meta['physics_dt'], meta['control_dt'], meta['substeps'], meta['history_length'], meta['goal_frames']) == (.005, .02, 4, 10, 11)
    layout = Layout(len(ck['joint_names']), len(ck['body_names']))
    assert layout.obs_dim == meta['layout']['obs_dim'] and layout.dim == meta['layout']['dim']
    actor = Actor(layout, ck['args']['small']).eval()
    actor.load_state_dict(ck['actor'], strict=True)
    for key in ('obs', 'goal'):
        norm = getattr(actor, key + '_norm')
        np.testing.assert_allclose(norm.mean.detach().numpy(), meta['normalizers'][key]['mean'], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(norm.std().detach().numpy(), meta['normalizers'][key]['std'], atol=1e-6, rtol=1e-6)
    os.environ[cfg['asset_env']] = str(xml)
    if robot == 'sih2300':
        os.environ[cfg['embodiment_contract']['yaml_env']] = str(Path(__file__).resolve().parents[2] / 'configs/supermimic/sih2300.yaml')
    model, params = build_model(cfg, ck['joint_names'])
    gains = [actuator_gains(p) for p in params]
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], [g[0] for g in gains])
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], [-g[1] for g in gains])
    scale = [.25 * p['effort'] / g[0] for p, g in zip(params, gains)]
    if native:
        meta['action_scale'] = scale
        assert model.opt.timestep == .005
        assert [model.joint(int(model.actuator_trnid[i,0])).name for i in range(model.nu)] == ck['joint_names']
    np.testing.assert_allclose(scale, meta['action_scale'])
    default = [sum(v for pattern, v in cfg['default_pose'].items() if re.fullmatch(pattern, name)) for name in ck['joint_names']]
    if native:
        meta['default_pose'] = default
    np.testing.assert_allclose(default, meta['default_pose'], atol=1e-6)
    output.mkdir(parents=True, exist_ok=False)
    motion = shared / f'datasets/lafan/motions/{robot}/walk3_subject3/motion.npz'
    with np.load(motion) as clip:
        assert float(clip['fps']) == 50
        assert clip['joint_names'].tolist() == ck['joint_names']
        state = clip['state'][1000:1511].copy()
    n = len(ck['joint_names'])
    # Root XYZ/WXYZ/world linear and angular velocity, then named joint q and qdot.
    reference = {'qpos': np.c_[state[:, :7], state[:, 13:13+n]],
                     'root_linear_velocity': state[:, 7:10], 'root_angular_velocity_world': state[:, 10:13],
                     'joint_velocity': state[:, 13+n:13+2*n], 'joint_names': ck['joint_names'], 'fps': 50, 'embodiment': robot}
    np.savez_compressed(output / 'reference50.npz', **reference)
    audit = {'robot': robot, 'checkpoint_sha256': entry['sha256'], 'contract_source': 'checkpoint_native' if native else 'web_export', 'iterations': entry['completed_iterations'], 'source_motion': str(motion),
                 'source_frames': [1000, 1511], 'strict_actor_load': True, 'normalizers_match': True,
                 'named_schema_match': True, 'xml_sha256': meta['source_asset_sha256'],
                 'pd_gains': gains, 'action_scale': meta['action_scale'], 'history_length': 10, 'goal_frames': 11,
                 'physics_hz': 200, 'tracker_hz': 50}
    (output / 'contract.json').write_text(json.dumps(audit, indent=2) + '\n')
    # Scene-free replay isolates Tracker behavior from Joystick and campus collisions.
    os.environ.pop('OMEGA_SUPERMIMIC_CONTACTS', None)
    result = run(robot, checkpoint, xml, output / 'reference50.npz', output, max_steps=500)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('robot', choices=tracker_release())
    parser.add_argument('contracts', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--shared', type=Path, default=Path(os.environ.get('OMEGA_SHARED_ROOT', Path.home()/'omega-assets')))
    args = parser.parse_args()
    verify(args.robot, None if str(args.contracts) == '-' else json.loads(args.contracts.read_text()), args.output, args.shared)
