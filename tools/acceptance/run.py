"""Run independent Tracker/Joystick gates and publish the five-layer acceptance matrix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from matrix import matrix

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--robots',nargs='+',default=['g1','a3','h2','sih2300'])
    p.add_argument('--stages',nargs='+',choices=['tracker','joystick','plane','render'],default=['tracker','joystick','plane'])
    p.add_argument('--render-python',default=sys.executable)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for stage in args.stages:
        for robot in args.robots:
            folder=args.output/({'joystick':'layer2','plane':'layer3'}.get(stage,'layer1'))/robot
            if stage=='render':
                if not (folder/'result.json').exists():continue
                command=[args.render_python,str(Path(__file__).with_name('render_plane.py')),str(folder)]
            else:
                if folder.exists():
                    raise FileExistsError(f'Refusing to overwrite evidence: {folder}')
                command=([sys.executable,str(Path(__file__).with_name('plane_loop.py')),'--robot',robot,'--root',str(args.output)]
                    if stage=='plane' else [sys.executable,str(Path(__file__).with_name('tracker_plane.py' if stage=='tracker' else 'joystick_reference.py')),
                    '--robot',robot,'--output',str(folder)])
            log=args.output/f'{stage}-{robot}.log'
            with log.open('w') as f:
                completed=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if completed.returncode:
                folder.mkdir(parents=True,exist_ok=True)
                (folder/('render-error.json' if stage=='render' else 'result.json')).write_text(json.dumps(
                    dict(robot=robot,status='error',exit_code=completed.returncode,log=str(log)),indent=2)+'\n')
            rows=matrix(args.output,['g1','a3','h2','sih2300'])
            (args.output/'matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(dict(stage=stage,robot=robot,returncode=completed.returncode)),flush=True)


if __name__=='__main__':main()
