"""Recorded-state contact QC distinguishes normal feet from scene obstacles."""
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('sdg_contacts', Path(__file__).parents[1]/'tools/sdg/check_contacts.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('robot,foot', [('g1','left_ankle_roll_link'), ('a3','left_ankle_roll_Link'), ('sih2300','LeftFoot')])
@pytest.mark.parametrize('scene', ['obstacle','campus'])
def test_floor_contact_is_allowed_but_obstacle_contact_fails(tmp_path, robot, foot, scene):
    xml = '''<mujoco><worldbody>
      <geom name="pg_0" type="plane" size="3 3 .1"/>
      <geom name="pg_1" type="box" pos="1 0 .3" size=".2 .2 .3"/>
      <body name="left_ankle_roll_link" pos="0 0 .09"><freejoint/>
        <geom name="foot" type="sphere" size=".1" mass="1"/>
        <body name="left_ankle_roll_sphere_1_link" pos=".04 0 0">
          <geom name="foot_sphere" type="sphere" size=".1" mass=".1"/>
        </body>
      </body></worldbody></mujoco>'''
    xml = xml.replace('left_ankle_roll_link', foot)
    if scene == 'campus':
        xml = xml.replace('name="pg_0"', 'name="pg_5003"').replace('</worldbody>', '<geom name="pg_5014" type="sphere" pos="30 30 0" size=".1"/></worldbody>')
    model = mujoco.MjModel.from_xml_string(xml)
    mujoco.mj_saveModel(model, str(tmp_path/'physics.mjb'), None)
    manifest = dict(control_steps=1, embodiment=dict(id=robot), scene=dict(id=scene, physical_collision_sha256=
        '754213e8a512ddffb988202916a58f6b2e9257ef9c5671f711e3ba48a409a8af' if scene == 'obstacle' else
        'a1f8f8087b1462f1cc977b3c0084df7946a296890f5fff4291bad44815965e4f'))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    np.savez(tmp_path/'physics_states.npz',qpos=model.qpos0[None],qvel=np.zeros((1,model.nv)),physics_step=[0])
    report=MODULE.check(tmp_path)
    assert report['passed'] and report['counts']['foot_ground'] > 0
    assert any(p['kind']=='foot_ground' and 'foot_sphere' in p['geoms'] for p in report['pairs'])
    q=model.qpos0.copy(); q[0]=.75
    np.savez(tmp_path/'physics_states.npz',qpos=q[None],qvel=np.zeros((1,model.nv)),physics_step=[0])
    report=MODULE.check(tmp_path)
    assert not report['passed'] and report['counts']['obstacle'] > 0


def test_candidate_diagnostics_cannot_publish(tmp_path):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="pg_0" type="plane" size="3 3 .1"/><body name="LeftFoot" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body><body name="RightFoot" pos="2 0 1"><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    mujoco.mj_saveModel(model, str(tmp_path/'physics.mjb'), None)
    manifest = dict(control_steps=1, embodiment=dict(id='sih2300'), scene=dict(id='obstacle', physical_collision_sha256='candidate'))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    np.savez(tmp_path/'physics_states.npz', qpos=model.qpos0[None], qvel=np.zeros((1,model.nv)), physics_step=[0])
    with pytest.raises(ValueError, match='pinned'):
        MODULE.check(tmp_path)
    report = MODULE.check(tmp_path, diagnostic_collision_sha256='candidate')
    assert report['diagnostic_geometry_passed'] and not report['passed']
    assert report['excluded_from_dataset']
    assert not (tmp_path/'contact_report.json').exists()
    assert (tmp_path/'diagnostic_contact_report.json').exists()


def test_audited_box_release_is_limited_to_sih(tmp_path):
    test_candidate_diagnostics_cannot_publish(tmp_path)
    p = tmp_path/'manifest.json'
    manifest = json.loads(p.read_text())
    manifest['scene']['physical_collision_sha256'] = MODULE.SIH_SUPPORT_BOX_SHA['obstacle']
    p.write_text(json.dumps(manifest))
    assert MODULE.check(tmp_path)['passed']
    manifest['embodiment']['id'] = 'h2'
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='pinned'):
        MODULE.check(tmp_path)


def test_plane_diagnostic_rejects_states_outside_original_support(tmp_path):
    test_candidate_diagnostics_cannot_publish(tmp_path)
    audit = tmp_path/'plane.audit.json'
    audit.write_text(json.dumps(dict(candidate_sha256='candidate', support_geom='pg_0', finite_support_bounds_tracker_xyz=[[-3,-3,-.2],[3,3,0]])))
    report = MODULE.check(tmp_path, diagnostic_collision_sha256='candidate', diagnostic_domain_audit=audit)
    assert report['diagnostic_geometry_passed'] and report['support_domain']['passed']
    assert not report['passed']
    with np.load(tmp_path/'physics_states.npz') as a:
        qpos, qvel, step = a['qpos'].copy(), a['qvel'], a['physics_step']
    qpos[0,0] = 4
    np.savez(tmp_path/'physics_states.npz', qpos=qpos, qvel=qvel, physics_step=step)
    report = MODULE.check(tmp_path, diagnostic_collision_sha256='candidate', diagnostic_domain_audit=audit)
    assert not report['support_domain']['passed'] and not report['diagnostic_geometry_passed']


@pytest.mark.parametrize('scene,ground', [('campus', 'pg_5003'), ('obstacle', 'pg_0')])
def test_h2_finite_heightfield_release_is_embodiment_specific(tmp_path, scene, ground):
    model = mujoco.MjModel.from_xml_string('<mujoco><asset><hfield name="floor" '
        'nrow="2" ncol="2" size="3 3 .1 .2" elevation="0 0 0 0"/></asset><worldbody>'
        f'<geom name="{ground}" type="hfield" hfield="floor"/>'
        '<body name="left_ankle_roll_link" pos="0 0 .09"><freejoint/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    mujoco.mj_saveModel(model, str(tmp_path/'physics.mjb'), None)
    manifest = dict(control_steps=1, embodiment=dict(id='h2'), scene=dict(id=scene,
        physical_collision_sha256=MODULE.H2_SUPPORT_HEIGHTFIELD_SHA[scene]))
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    np.savez(tmp_path/'physics_states.npz', qpos=model.qpos0[None],
             qvel=np.zeros((1, model.nv)), physics_step=[0])
    report = MODULE.check(tmp_path)
    assert report['passed'] and report['counts']['foot_ground'] > 0
    assert report['allowed_support_geoms'] == [ground]
    manifest['embodiment']['id'] = 'g1'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='pinned'):
        MODULE.check(tmp_path)
