"""Render actual stored SuperMimic link poses against the matching scene's GPU visual bundle."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from omega_playground.ops.scenes import SCENES
from omega_playground.scene.package import load
from omega_playground.viewer.mjwarp_live import MjWarpViewer


def render(job, robot, scene_key, xml):
    visual = Path('artifacts/mjwarp')/f'supermimic-{robot}-{scene_key}'
    meta = json.loads((visual/'asset.json').read_text())
    scene_name, _, config = SCENES[scene_key]
    scene_root = Path('artifacts/scenes')/scene_name
    if scene_key == 'campus' and not scene_root.exists():
        scene_root = Path('artifacts/scenes/VibeCampus-v20261013')
    scene = load(scene_root)
    if (meta['schema'] != 'omega.supermimic-visual/1' or meta['scene_sha256'] != scene.scene_sha256
            or meta['xml_sha256'] != hashlib.sha256(xml.read_bytes()).hexdigest()):
        raise ValueError('Prepared visual does not match this robot XML and scene')
    from xml.etree import ElementTree as ET
    contacts = ET.parse(Path.cwd().parent/f'{scene_key}-full-contacts-v2.xml')
    elevation = -float(contacts.find('worldbody/geom').get('pos', '0 0 0').split()[2])
    origin = json.loads((Path('configs/scenes')/config).read_text())['route'][0] + [elevation]
    model = mujoco.MjModel.from_binary_path(str(visual/meta['entry']))
    with np.load(job/'body_states.npz', allow_pickle=False) as archive:
        names = archive['body_names'].tolist()
        order = [names.index(name) for name in meta['body_names']]
        poses = archive['body_q'][:,order].copy()
        poses[:,:,:3] += origin
        times = archive['sim_time']
    temporary = job/'replay.pending.mp4'
    viewer = MjWarpViewer(model,meta,visual_directory=visual,width=960,height=540,video=temporary,camera='chase',video_bitrate_kbps=4000)
    ego = None
    if viewer.ego_body is not None:
        ego = subprocess.Popen(['ffmpeg','-v','error','-y','-f','mjpeg','-r','25','-i','pipe:0',
                                '-an','-c:v','hevc_nvenc','-tag:v','hvc1','-pix_fmt','yuv420p',
                                '-movflags','+faststart',str(job/'ego.pending.mp4')],stdin=subprocess.PIPE)
    def encode(rgb, record):
        viewer.encode_frame(rgb,record)
        if ego:
            ego.stdin.write(viewer.ego_jpeg)
    viewer.frame_sink = encode
    try:
        for i in range(0,len(poses),2):
            viewer.show(poses[i],float(times[i]))
            if i == 0:
                (job/'preview.jpg').write_bytes(viewer.latest_jpeg)
    finally:
        try:
            viewer.close()
        finally:
            if ego:
                ego.stdin.close()
                if ego.wait() != 0:
                    raise RuntimeError('Ego video encoder failed')
    if ego:
        (job/'ego.pending.mp4').replace(job/'ego.mp4')
    temporary.replace(job/'replay.mp4')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job',type=Path)
    p.add_argument('robot')
    p.add_argument('scene',choices=SCENES)
    p.add_argument('xml',type=Path)
    args = p.parse_args()
    render(args.job,args.robot,args.scene,args.xml)
