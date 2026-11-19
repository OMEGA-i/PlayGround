"""Replay authoritative poses through the Web renderer worker and H265 output."""
import argparse
import json
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from omega_playground.policy.package import load
from omega_playground.viewer.process import RenderProcess, mjwarp_scene_viewer


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--visual-model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--render-python', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    package = load(args.package)
    report = json.loads((args.input/'report.json').read_text())
    poses = np.load(args.input/'body_poses.npy', mmap_mode='r')
    if report['body_order'] != package.body_names:
        raise ValueError('Policy pose order mismatch')
    received = []
    factory = partial(mjwarp_scene_viewer, str(args.package), str(args.visual_model),
                      report['scene']['scene_sha256'], report['num_agents'], 'cuda:0',
                      video=args.output/'stream.mp4', video_codec='h265', profile=True)
    def frame(jpeg, sim_time, record, wall):
        (args.output/'latest.jpg').write_bytes(jpeg)
        if record:
            received.append((sim_time, wall))
    viewer = RenderProcess(factory, frame, python_executable=args.render_python)
    anchor = package.body_names.index(package.robot['anchor_body'])
    viewer.root_bodies = [i*len(package.body_names)+anchor for i in range(report['num_agents'])]
    viewer.follow_body = anchor
    viewer.camera_mode = 'squad'
    first = round(args.start/report['physics_dt'])
    stride = round(1/(25*report['physics_dt']))
    start = time.perf_counter()
    try:
        viewer.show(poses[first], args.start, record=False)
        start = time.perf_counter()
        for index in range(args.frames):
            step = first+index*stride
            viewer.show(poses[step], step*report['physics_dt'])
    finally:
        result = viewer.close()
    result['wall_seconds'] = time.perf_counter()-start
    result['received_frames'] = len(received)
    result['physics_steps'] = 0
    result['host_python'] = sys.executable
    probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=codec_name,nb_read_frames,r_frame_rate', '-of', 'json', str(args.output/'stream.mp4')], text=True)
    result['video_probe'] = json.loads(probe)
    result['passed'] = (len(received) == args.frames and not result['newton_imported']
                        and result['video_probe']['streams'][0]['codec_name'] == 'hevc'
                        and int(result['video_probe']['streams'][0]['nb_read_frames']) == args.frames)
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise RuntimeError('Renderer process/H265 acceptance failed')


if __name__ == '__main__':
    main()
