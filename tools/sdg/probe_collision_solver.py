"""Isolate native-CCD effects using the unchanged batched task recorder.

This diagnostic only changes the collision algorithm. The exact compiled model
is saved by run_batch; inputs must explicitly declare the experimental setting.
Outputs stay outside the production raw queue.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    task = json.loads(args.task.read_text())
    if task.get('diagnostic_physics_options') != {'nativeccd': False}:
        raise ValueError('Task must declare diagnostic_physics_options.nativeccd=false')
    if 'probes' not in args.output.parts or 'raw' in args.output.parts:
        raise ValueError('Solver diagnostics must be isolated in probes, outside raw production')
    import supermimic.mjwarp as runtime
    original = runtime.build_model

    def diagnostic_model(cfg, names, scene_xml=None):
        return original(dict(cfg, mujoco_nativeccd=False), names, scene_xml=scene_xml)

    runtime.build_model = diagnostic_model
    try:
        from run_batch import main as run_batch
        run_batch()
    finally:
        runtime.build_model = original


if __name__ == '__main__':
    main()
