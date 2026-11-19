"""Add a truthful post-run provenance supplement without rewriting execution metadata."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from omega_playground.policy.supermimic.provenance import file_identity, repository, runtime

ROOT=Path(__file__).resolve().parents[2]


def audit(execution,shared):
    manifest=json.loads((execution/'manifest.json').read_text())
    commit=manifest['versions']['playground']
    def at_commit(path):
        return subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{path}'])
    pins=json.loads(at_commit('configs/supermimic/joysticks.json'))[manifest['embodiment']['id']]
    loader=at_commit('src/omega_playground/policy/supermimic/live_planner.py').decode()
    worker=at_commit('tools/supermimic/supermimic_joystick_worker.py').decode()
    joystick_seed=7 if 'torch.manual_seed(7)' in worker else None
    if "sha256(tokenizer) != pins['tokenizer']['sha256']" not in loader or "sha256(path) != pins[stage]['sha256']" not in loader:
        raise ValueError('Recorded code revision does not prove checkpoint hash enforcement')
    checkpoints={}
    for name,pin in pins.items():
        identity=file_identity(shared/'joystick'/pin['path'])
        if identity['sha256']!=pin['sha256']:
            raise ValueError(f'Current {name} checkpoint differs from execution pin')
        checkpoints[name]=dict(**identity,expected_sha256=pin['sha256'],evidence='execution revision validates SHA256 before model loading; current bytes match')
    trackers=json.loads(at_commit('configs/supermimic/trackers.json'))['checkpoints']
    tracker=next(row for row in trackers if row['robot']==manifest['embodiment']['id'])
    identity=file_identity(shared/tracker['relative_path'])
    if identity['sha256']!=manifest['versions']['tracker_sha256']:
        raise ValueError('Current Tracker checkpoint differs from recorded execution hash')
    checkpoints['tracker']=identity
    # The physics model file was saved at execution initialization; this is a file
    # timestamp anchor, not a fabricated process-start timestamp.
    anchor=(execution/'physics.mjb').stat().st_mtime
    repos={}
    for name in ('Joystick','SuperMimicKit'):
        directory=ROOT.parent/name
        value=repository(directory)
        reflog=directory/'.git/logs/HEAD'
        value['head_reflog_mtime']=reflog.stat().st_mtime if reflog.is_file() else None
        value['head_reflog_predates_model_save']=bool(reflog.is_file() and reflog.stat().st_mtime<anchor)
        value['evidence_scope']='post-run clean checkout and HEAD reflog; not an execution-time code snapshot'
        repos[name]=value
    interpreters={'physics':Path(os.environ.get('OMEGA_SUPERMIMIC_PYTHON',ROOT.parent/'SuperMimicKit/.venv/bin/python')),
        'joystick':Path(os.environ.get('OMEGA_JOYSTICK_PYTHON',ROOT.parent/'Joystick/.venv/bin/python')),
        'render':Path(os.environ.get('OMEGA_RENDER_PYTHON',ROOT.parent/'venvs/pg-mjwarp-clean-001/bin/python'))}
    environments={}
    for role,python in interpreters.items():
        command='import json; from omega_playground.policy.supermimic.provenance import runtime; print(json.dumps(runtime()))'
        environments[role]=json.loads(subprocess.check_output([str(python),'-c',command],env={**os.environ,'PYTHONPATH':str(ROOT/'src')},text=True))
    report=dict(schema='omega.playground.provenance-audit/1',sdg_task_uid=manifest['sdg_task_uid'],
        audit_time=datetime.now(timezone.utc).isoformat(),execution_playground_commit=commit,
        manifest_identity=file_identity(execution/'manifest.json'),checkpoints=checkpoints,
        seeds=dict(task=manifest['seed'],joystick=joystick_seed,joystick_seed_evidence='literal manual_seed in recorded worker revision'),
        observed_repositories=repos,observed_environments=environments,physics_model_saved_mtime=anchor,
        limitations=['Runtime library versions and Joystick/SuperMimicKit revisions were not captured during this execution.',
            'Environment values above are explicitly post-run observations, not retroactively asserted execution-time versions.'],
        model_provenance_verified=True)
    (execution/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(sdg_task_uid=report['sdg_task_uid'],model_provenance_verified=True,
        output=str(execution/'provenance.json'),limitations=report['limitations'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--shared',type=Path,default=Path(os.environ.get('OMEGA_SHARED_ROOT',Path.home()/'omega-assets')))
    args=parser.parse_args()
    audit(args.execution,args.shared)
