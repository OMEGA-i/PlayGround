"""Private inherited socket: native 25 Hz Joystick and rolling 50 Hz joint references."""
import os
import copy
import hashlib
import argparse
import json
from multiprocessing.connection import Connection
from pathlib import Path
import traceback

import torch

from omega_playground.policy.supermimic.live_planner import RollingJoystick


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fd',type=int)
    p.add_argument('robot')
    p.add_argument('xml',type=Path)
    args=p.parse_args()
    pg=Path(__file__).resolve().parents[2]
    conn=Connection(args.fd)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(7)
        pins=json.loads((pg/'configs/supermimic/joysticks.json').read_text())[args.robot]
        with torch.inference_mode():
            planner=RollingJoystick(pg.parent/'Joystick',(Path(os.environ.get('OMEGA_SHARED_ROOT', Path.home()/'omega-assets'))/'joystick'),args.robot,args.xml,pins)
            from omega_playground.policy.supermimic.provenance import runtime, repository, file_identity
            assets=Path(os.environ.get('OMEGA_SHARED_ROOT', Path.home()/'omega-assets'))/'joystick'
            provenance=dict(runtime=runtime(),code=repository(pg.parent/'Joystick'),seed=7,
                model_hash_validation='all three checkpoint SHA256 values verified before loading',
                checkpoints={key:file_identity(assets/value['path']) for key,value in pins.items()},
                primitive_bank=file_identity(pg.parent/'Joystick/configs/soma_primitives_25fps.json'))
            conn.send({'ready':True,'provenance':provenance})
            streams = {}
            while True:
                command=conn.recv()
                if command is None:break
                if 'batch' in command:
                    selected, commands = [], []
                    for request in command['batch']:
                        stream = request['env_id']
                        if stream not in streams:
                            streams[stream] = copy.copy(planner)
                            streams[stream].reset()
                        current = streams[stream]
                        if request.get('reset'):
                            current.reset()
                        selected.append(current)
                        commands.append((*request['velocity'], request.get('observation')))
                    batch_seed = int.from_bytes(hashlib.sha256(json.dumps([r.get('seed', 7) for r in command['batch']]).encode()).digest()[:4], 'little')
                    torch.manual_seed(batch_seed)
                    references = RollingJoystick.release_many(selected, commands)
                    for reference in references:
                        reference['sampling_batch_seed'] = batch_seed
                        reference['sampling_batch_env_ids'] = [r['env_id'] for r in command['batch']]
                    conn.send({'references': references})
                    continue
                if command.get('reset'):
                    planner.reset()
                conn.send(planner.release(*command['velocity'],observation=command.get('observation')))
    except EOFError:
        pass
    except BaseException:
        conn.send({'error':traceback.format_exc()})
        raise
    finally:
        conn.close()


if __name__=='__main__':
    main()
