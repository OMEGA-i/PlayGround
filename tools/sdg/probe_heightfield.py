"""Load diagnostic heightfield scene assets without changing shared Tracker code.

The adapter only makes the scene's hfield declarations available to the existing
model assembler. The complete compiled physics model remains the recorded truth.
"""
import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    task = json.loads(args.task.read_text())
    if task.get('diagnostic_scene_assets') != ['hfield']:
        raise ValueError('Task must declare its diagnostic hfield adapter')
    if 'probes' not in args.output.parts or 'raw' in args.output.parts:
        raise ValueError('Heightfield diagnostics must stay outside raw production')
    shared = Path(os.environ['OMEGA_SHARED_ROOT'])
    from run_episode import collision_asset
    scene = ET.parse(collision_asset(task, shared))
    if not scene.findall('asset/hfield'):
        raise ValueError('Scene contains no heightfields')
    import supermimic.mjwarp as runtime
    original = runtime.stage_physx_asset

    def stage(source, destination):
        path = original(source, destination)
        tree = ET.parse(path)
        assets = tree.find('asset')
        if assets is None:
            assets = ET.SubElement(tree.getroot(), 'asset')
        assets.extend(scene.findall('asset/hfield'))
        tree.write(path, encoding='unicode')
        return path

    runtime.stage_physx_asset = stage
    try:
        from run_batch import main as run_batch
        run_batch()
    finally:
        runtime.stage_physx_asset = original


if __name__ == '__main__':
    main()
