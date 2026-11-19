"""Recompute geometric contacts at every saved control state with the exact physics model.

This is 50 Hz collision evidence, not a claim about unrecorded 200 Hz substeps or
contact forces. No dynamics are advanced and no policy is re-executed.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import mujoco
import numpy as np


# Audited native boxes preserve the released support volumes exactly.
# These releases have been validated for SIH2300 only.
SIH_SUPPORT_BOX_SHA = {
    'campus': '74a2d3b042d3b08eb897a538d999ac0aea936db20af26f3f950174a7666bc794',
    'obstacle': 'd2279b32630e2a2c8ebde6b4504078f5ed7661535f8011bed1b36316642e2ded',
}


# Finite 0.1 m flat heightfield: audited support union, unchanged obstacles.
# H2: independent diagnostic batches plus native-entry validation; failed tasks excluded.
H2_SUPPORT_HEIGHTFIELD_SHA = {
    'campus': '0893efb230ec0571c33e521437a5defd1c808a3cd2fea1805038627941a524e7',
    'obstacle': '76b74524dfc6e1d5bd2810de31da2586834b6a9c46d1ad490fe52efd603a98e2',
}


def check(execution, *, diagnostic_collision_sha256=None, diagnostic_domain_audit=None):
    model = mujoco.MjModel.from_binary_path(str(execution/'physics.mjb'))
    data = mujoco.MjData(model)
    root = int(model.jnt_bodyid[0])
    is_robot = np.asarray(model.body_rootid) == root
    manifest = json.loads((execution/'manifest.json').read_text())
    profiles = {
        'obstacle': ('754213e8a512ddffb988202916a58f6b2e9257ef9c5671f711e3ba48a409a8af', ('pg_0',)),
        'campus': ('a1f8f8087b1462f1cc977b3c0084df7946a296890f5fff4291bad44815965e4f', ('pg_5003', 'pg_5014')),
    }
    digest, ground_names = profiles[manifest['scene']['id']]
    accepted = {digest}
    if manifest['embodiment']['id'] == 'sih2300':
        accepted.add(SIH_SUPPORT_BOX_SHA[manifest['scene']['id']])
    h2_heightfield = (manifest['embodiment']['id'] == 'h2'
        and manifest['scene'].get('physical_collision_sha256')
        == H2_SUPPORT_HEIGHTFIELD_SHA.get(manifest['scene']['id'], 'unreleased'))
    if h2_heightfield:
        accepted.add(H2_SUPPORT_HEIGHTFIELD_SHA[manifest['scene']['id']])
        ground_names = (ground_names[0],)
    if diagnostic_collision_sha256 is not None:
        accepted = {diagnostic_collision_sha256}
    if manifest['scene'].get('physical_collision_sha256') not in accepted:
        raise ValueError('Ground classification requires the pinned collision release')
    domain = None
    if diagnostic_domain_audit is not None:
        domain = json.loads(Path(diagnostic_domain_audit).read_text())
        if not diagnostic_collision_sha256 or domain['candidate_sha256'] != diagnostic_collision_sha256:
            raise ValueError('Diagnostic support domain must match the explicit collision hash')
        ground_names = (domain['support_geom'],)
    ground = {model.geom(name).id for name in ground_names}
    robot_geoms = np.flatnonzero(is_robot[model.geom_bodyid] & ((model.geom_contype | model.geom_conaffinity) != 0))
    minimum_domain_margin = float('inf')
    maximum_foot_penetration = 0.
    feet = {
        'g1': ('left_ankle_roll_link', 'right_ankle_roll_link'),
        'h2': ('left_ankle_roll_link', 'right_ankle_roll_link'),
        'a3': ('left_ankle_roll_Link', 'right_ankle_roll_Link'),
        'sih2300': ('LeftFoot', 'RightFoot'),
    }[manifest['embodiment']['id']]
    def in_foot_subtree(body_id):
        while body_id:
            if model.body(body_id).name in feet:
                return True
            body_id = int(model.body_parentid[body_id])
        return False
    contacts, pairs = [], Counter()
    counts = Counter()
    with np.load(execution/'physics_states.npz') as archive:
        positions, velocities, steps = archive['qpos'], archive['qvel'], archive['physics_step']
        if len(positions) != manifest['control_steps'] + (1 if manifest.get('recording_body_hz') == 50 else 0) or not np.array_equal(steps, np.arange(len(positions))*4):
            raise ValueError('Missing control-state collision evidence')
        for qpos, qvel, step in zip(positions, velocities, steps):
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)
            if domain is not None:
                low, high = np.asarray(domain['finite_support_bounds_tracker_xyz'])
                centers = data.geom_xpos[robot_geoms, :2]
                radii = model.geom_rbound[robot_geoms, None]
                margin = float(np.minimum(centers-low[:2]-radii, high[:2]-centers-radii).min())
                minimum_domain_margin = min(minimum_domain_margin, margin)
            for c in data.contact[:data.ncon]:
                if c.dist > 0:
                    continue
                a,b = map(int, c.geom)
                ra,rb = bool(is_robot[model.geom_bodyid[a]]),bool(is_robot[model.geom_bodyid[b]])
                if not ra and not rb:
                    continue
                if ra and rb:
                    kind = 'self_contact'
                else:
                    robot, scene = (a,b) if ra else (b,a)
                    foot = in_foot_subtree(int(model.geom_bodyid[robot]))
                    kind = 'foot_ground' if foot and scene in ground else ('nonfoot_ground' if scene in ground else 'obstacle')
                names = [model.geom(g).name or f'geom_{g}' for g in (a,b)]
                if kind == 'foot_ground':
                    maximum_foot_penetration = max(maximum_foot_penetration, float(-c.dist))
                counts[kind] += 1
                pairs[(kind,*names)] += 1
                if kind != 'foot_ground':
                    contacts.append(dict(physics_step=int(step),timestamp=float(step)/200,kind=kind,
                        geoms=names,penetration_m=float(-c.dist),position=c.pos.tolist()))
    report = dict(schema='omega.playground.contact-qc/1', method='mujoco_forward_on_saved_physics_states',
        physical_collision_sha256=manifest['scene']['physical_collision_sha256'],
        sample_hz=50, physics_hz=200, checked_control_states=len(positions), substep_contacts='not_recorded',
        acceptance_scope='robot versus scene; self-contact reported separately',
        allowed_support_geoms=list(ground_names), foot_subtree_roots=list(feet),
        self_contact_count=counts['self_contact'],
        max_self_penetration_m=max((c['penetration_m'] for c in contacts if c['kind']=='self_contact'),default=0.),
        max_foot_ground_penetration_m=maximum_foot_penetration,
        counts=dict(counts), unexpected_scene_contact_count=counts['obstacle']+counts['nonfoot_ground'],
        passed=counts['obstacle']+counts['nonfoot_ground']==0,
        pairs=[dict(kind=k[0],geoms=list(k[1:]),count=v) for k,v in sorted(pairs.items())],
        unexpected_contacts=contacts)
    filename = 'contact_report.json'
    if diagnostic_collision_sha256 is not None:
        if domain is not None:
            low, high = np.asarray(domain['finite_support_bounds_tracker_xyz'])
            domain_passed = minimum_domain_margin >= 0 and maximum_foot_penetration <= high[2]-low[2]
            report['support_domain'] = dict(passed=bool(domain_passed),
                minimum_horizontal_bounding_sphere_margin_m=minimum_domain_margin,
                maximum_allowed_penetration_m=float(high[2]-low[2]), sample_hz=50)
            report['passed'] = bool(report['passed'] and domain_passed)
        report['diagnostic_geometry_passed'] = report['passed']
        report['passed'] = False
        report['excluded_from_dataset'] = True
        report['physical_collision_sha256'] = diagnostic_collision_sha256
        filename = 'diagnostic_contact_report.json'
    temporary = execution/f'.{filename}.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(execution/filename)
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','unexpected_contacts')}))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--diagnostic-collision-sha256', help='Measure a pinned candidate, always excluded from dataset acceptance')
    parser.add_argument('--diagnostic-domain-audit', type=Path)
    args=parser.parse_args()
    raise SystemExit(0 if check(args.execution, diagnostic_collision_sha256=args.diagnostic_collision_sha256, diagnostic_domain_audit=args.diagnostic_domain_audit)['passed'] else 2)
