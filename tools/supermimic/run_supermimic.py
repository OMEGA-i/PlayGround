"""Run the independent SuperMimic workflow with the two owner environments."""
import argparse
import json
import math
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
from omega_playground.policy.supermimic.release import tracker_checkpoint, tracker_release

STEPS = {robot: entry['completed_iterations'] for robot, entry in tracker_release().items()}
SHARED = Path(os.environ.get('OMEGA_SHARED_ROOT', Path.home()/'omega-assets'))
WORKSPACE = ROOT.parent


def velocity_command(values):
    if len(values) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Expected finite vx, vy')
    vx, vy = values
    if not -1 <= vx <= 2 or not -1 <= vy <= 1:
        raise ValueError('vx must be within [-1, 2] and vy within [-1, 1] m/s')
    return vx, vy


def execute(robot, output, velocity=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    joystick = WORKSPACE/'Joystick'
    tracker = WORKSPACE/'SuperMimicKit'
    xml = (SHARED/'joystick/assets/sih2300/robot.xml' if robot == 'sih2300' else
           SHARED/'supermimic/assets/robots'/robot/'robot.xml')
    ck = tracker_checkpoint(robot, SHARED)
    env = {**os.environ, 'PYTHONPATH':str(ROOT/'src')}
    def stage(python, args, cwd):
        subprocess.run([str(python), *map(str,args)], cwd=cwd, env=env, check=True)
    planner = Path(os.environ.get('OMEGA_JOYSTICK_PYTHON', joystick/'.venv/bin/python'))
    reference = output/'joystick'
    conditions = []
    if velocity is not None:
        vx, vy = velocity_command(velocity)
        # Use the Joystick owner's exact directional primitive selection.
        category = subprocess.check_output([str(planner), '-c',
            'from joystick.web_generate import category; import sys; print(category(*map(float,sys.argv[1:])))',
            str(vx), str(vy)], cwd=joystick, env={**env, 'PYTHONPATH':str(joystick/'src')}, text=True).strip()
        conditions = ['--command', str(vx), str(vy), '--category', category]
    stage(planner, [joystick/'tools/offline_generation.py','--base',SHARED/'joystick',
          '--robot',robot,'--xml',xml,'--output',reference,'--condition-mode','sonic',
          '--export-tracker-reference', *conditions], joystick)
    expected = json.loads((ROOT/'configs/supermimic/joysticks.json').read_text())[robot]
    observed = json.loads((reference/'report.json').read_text())['checkpoints']
    for name, pin in expected.items():
        if observed[name]['sha256'] != pin['sha256']:
            raise ValueError(f'{robot} {name} differs from the Joystick delivery')
    stage(planner, [__file__, 'convert',robot,output,xml,
                    *([] if velocity is None else ['--velocity', *map(str, velocity)])], ROOT)
    stage(Path(os.environ.get('OMEGA_SUPERMIMIC_PYTHON', tracker/'.venv/bin/python')), [__file__, 'track',robot,output,xml,ck], tracker)
    scene_key = os.environ.get('OMEGA_SUPERMIMIC_SCENE')
    if scene_key in ('campus', 'obstacle') and (ROOT/'artifacts/mjwarp'/f'supermimic-{robot}-{scene_key}'/'asset.json').exists():
        stage(Path(os.environ.get('OMEGA_RENDER_PYTHON', WORKSPACE/'venvs/pg-mjwarp-clean-001/bin/python')),
              [ROOT/'tools/supermimic/render_supermimic.py',output,robot,scene_key,xml],ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['run','convert','track','diagnose'])
    p.add_argument('robot',choices=STEPS)
    p.add_argument('output',type=Path)
    p.add_argument('paths',nargs='*',type=Path)
    p.add_argument('--velocity', type=float, nargs=2, metavar=('VX','VY'))
    p.add_argument('--steps', type=int, default=500, help='Maximum diagnostic Tracker steps at 50 Hz')
    p.add_argument('--scenarios', nargs='+', choices=('recorded','forward','idle_forward','idle','auto','auto_turn',
                                                   'source_forward','source_left','source_idle'))
    args = p.parse_args()
    reference = args.output/'reference50.npz'
    if args.stage == 'diagnose':
        if args.steps < 1:
            p.error('Diagnostic steps must be positive')
        if args.scenarios and any(s.startswith('auto') for s in args.scenarios) and len(args.scenarios)!=1:
            p.error('Auto replay must run by itself')
        from omega_playground.policy.supermimic.diagnostic import diagnose
        diagnose(args.robot, args.output, SHARED, ROOT, args.scenarios, args.steps)
    elif args.stage == 'run':
        execute(args.robot,args.output,args.velocity)
    elif args.stage == 'convert':
        import numpy as np
        from omega_playground.policy.supermimic.reference import native_reference
        np.savez_compressed(reference,**native_reference(args.output/'joystick'/('custom_rolling.npz' if args.velocity is not None else 'left_rolling.npz'),
                                                        args.paths[0],args.robot))
    else:
        from omega_playground.policy.supermimic.rollout import run
        print(json.dumps(run(args.robot,args.paths[1],args.paths[0],reference,args.output)))


if __name__ == '__main__':
    main()
