"""Render a read-only third-person review of a sealed physical execution."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from render_episode import Encoder, ROOT
from omega_playground.viewer.mjwarp_live import MjWarpViewer
from omega_playground.recording.execution import camera_samples


def render(source, output):
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['status'] not in ('complete', 'failed', 'aborted') or manifest.get('execution_error'):
        raise ValueError('A sealed execution without capture errors is required')
    robot, scene = manifest['embodiment']['id'], manifest['scene']['id']
    visual = ROOT / 'artifacts/mjwarp' / f'supermimic-{robot}-{scene}'
    meta = json.loads((visual / 'asset.json').read_text())
    if (meta['xml_sha256'] != manifest['embodiment']['model_sha256']
            or meta['scene_sha256'] != manifest['scene']['sha256']):
        raise ValueError('Visual bundle differs from physical execution')
    transform = np.asarray(manifest['scene_from_tracker'])
    if not np.allclose(transform[:3, :3], np.eye(3)):
        raise ValueError('Unsupported scene rotation')
    with np.load(source / 'body_states.npz', allow_pickle=False) as data:
        order = [data['body_names'].tolist().index(name) for name in meta['body_names']]
        indexes = camera_samples(data['control_tick'], manifest['control_steps'])
        poses = data['body_q'][indexes][:, order].copy()
    poses[:, :, :3] += transform[:3, 3]
    output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_binary_path(str(visual / meta['entry']))
    viewer = MjWarpViewer(model, meta, visual_directory=visual, width=1280, height=720, camera='chase')
    pending = output / 'main.pending.mp4'
    encoder = Encoder(pending, 1280, 720)
    try:
        for i, body in enumerate(poses):
            target = body[viewer.follow_body, :3].copy()
            viewer.renderer.look_at(target + [-4., -3., 2.], target)
            rgb = viewer.rgb_readback.read(viewer.renderer.render(body))[0].copy()
            encoder.write(rgb)
            if i in (0, len(poses)//2, len(poses)-1):
                Image.fromarray(rgb).save(output / f'frame-{i:06d}.jpg')
            if i % 100 == 0:
                print(f'{robot}: {i+1}/{len(poses)}', flush=True)
    finally:
        encoder.close()
        viewer.close()
    pending.replace(output / 'main.mp4')
    (output / 'source.json').write_text(json.dumps(dict(source=str(source),
        embodiment=robot, scene=scene, frames=len(poses), fps=25,
        end_reason=manifest.get('end_reason'), tracker=manifest['versions']['tracker_checkpoint'],
        task=manifest['task'], source_type='actual_physics_body_poses'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.source, args.output)
