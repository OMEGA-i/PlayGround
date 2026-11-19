"""Render paired RGB/forward-depth from the recorded actual physical body poses."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from omega_playground.viewer.mjwarp_live import MjWarpViewer
from omega_playground.recording.execution import camera_samples
from omega_playground.policy.supermimic.provenance import runtime, repository, file_identity

ROOT = Path(__file__).resolve().parents[2]
_RENDER_CACHE = {}


class Encoder:
    def __init__(self, path, width, height, depth=False, fps=25):
        self.proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt',
            'gray16le' if depth else 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an',
            *(['-c:v','ffv1','-level','3','-g','1','-pix_fmt','gray16le'] if depth else
              ['-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart']),
            str(path)], stdin=subprocess.PIPE)
    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError('Video encoder failed')


def render(output, width=640, height=480, *, reuse=False, replay_stride=1):
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest.get('execution_error') or manifest['status'] not in ('complete', 'failed', 'aborted'):
        raise ValueError('Execution must be sealed and free of capture errors before rendering')
    if manifest['control_steps'] < 1:
        raise ValueError('Cannot render a task without control intervals')
    robot, scene = manifest['embodiment']['id'], manifest['scene']['id']
    if robot not in ('g1', 'a3', 'h2', 'sih2300') or scene not in ('campus', 'obstacle'):
        raise ValueError('Unknown released render profile')
    visual = ROOT/'artifacts/mjwarp'/f'supermimic-{robot}-{scene}'
    meta = json.loads((visual/'asset.json').read_text())
    if meta['xml_sha256'] != manifest['embodiment']['model_sha256'] or meta['scene_sha256'] != manifest['scene']['sha256']:
        raise ValueError('Visual bundle differs from executed robot/scene')
    cache_key = (str(visual), width, height)
    cached = _RENDER_CACHE.get(cache_key) if reuse else None
    model = cached[0] if cached else mujoco.MjModel.from_binary_path(str(visual/meta['entry']))
    with np.load(output/'body_states.npz') as a:
        order = [a['body_names'].tolist().index(name) for name in meta['body_names']]
        poses = a['body_q'][:,order].copy()
        ticks = a['control_tick'].copy()
    indexes = camera_samples(ticks, manifest['control_steps'])
    poses, ticks = poses[indexes], ticks[indexes]
    transform = np.asarray(manifest['scene_from_tracker'])
    poses[:,:,:3] += transform[:3,3]
    if not np.allclose(transform[:3,:3], np.eye(3)):
        raise ValueError('Unexpected scene rotation; explicit pose transform required')
    sensors = output/'sensors'
    sensors.mkdir(exist_ok=True)
    viewer = cached[1] if cached else MjWarpViewer(model, meta, visual_directory=visual, width=width, height=height, camera='chase')
    if reuse:
        _RENDER_CACHE[cache_key] = (model, viewer)
    if viewer.supersample != 1 or viewer.edge_antialias:
        raise ValueError('SDG RGB-D requires a shared native pixel grid; disable supersampling and edge filtering')
    renderer = viewer.renderer
    if viewer.ego_body is None:
        raise ValueError('Embodiment optical mount missing')
    encoders = [Encoder(sensors/'ego.rgb.mp4', width,height), Encoder(sensors/'ego.depth.mkv',width,height,True),
                Encoder(output/'replay.mp4',width,height,fps=25/replay_stride)]
    depth_hashes, camera_poses = [], []
    try:
        for i, body in enumerate(poses):
            pose = body[viewer.ego_body]
            orientation = Rotation.from_quat(pose[3:7])
            eye = pose[:3] + orientation.apply(viewer.ego_offset)
            forward = orientation.apply(viewer.ego_forward)
            renderer.look_at(eye, eye+forward)
            rgb = viewer.rgb_readback.read(renderer.render(body))[0].copy()
            # The shared renderer writes planar optical-axis depth for this same ray dispatch.
            depth = renderer.context.depth_data.numpy()[0,:width*height].reshape(height,width)
            scaled = np.rint(depth.astype(np.float64)/.001)
            valid = np.isfinite(depth)&(depth>0)&(depth<=65.535)&(scaled>=1)&(scaled<=65535)
            codes = np.zeros(depth.shape, dtype='<u2')
            codes[valid] = scaled[valid].astype('<u2')
            encoders[0].write(rgb)
            encoders[1].write(codes)
            depth_hashes.append(hashlib.sha256(codes.tobytes()).hexdigest())
            z = forward/np.linalg.norm(forward)
            x = np.cross(z,[0.,0.,1.]); x /= np.linalg.norm(x)
            y = np.cross(z,x)
            optical = np.eye(4); optical[:3,:3]=np.stack((x,y,z),axis=1); optical[:3,3]=eye
            camera_poses.append(optical)
            # A separate third-person view documents the actual traversal.
            target = body[viewer.follow_body,:3].copy()
            if i % replay_stride == 0 or i in {0,len(poses)//2,len(poses)-1}:
                renderer.look_at(target+[-4.,-3.,2.], target)
                exo = viewer.rgb_readback.read(renderer.render(body))[0].copy()
                if i % replay_stride == 0:
                    encoders[2].write(exo)
            if i in {0,len(poses)//2,len(poses)-1}:
                Image.fromarray(rgb).save(output/f'ego-{i:06d}.png')
                Image.fromarray(exo).save(output/f'exo-{i:06d}.png')
                Image.fromarray(codes).save(output/f'depth-{i:06d}.png')
                # Fixed 2.5 m vertical plane through the root, camera level at 1.25 m.
                qc_target = target.copy(); qc_target[2] = 1.25
                distance = 1.25/math.tan(math.radians(float(model.cam_fovy[0]))/2)
                renderer.look_at(qc_target+np.array([-.8660254037844386,-.5,0.])*distance, qc_target)
                qc = Image.fromarray(viewer.rgb_readback.read(renderer.render(body))[0].copy())
                draw = ImageDraw.Draw(qc)
                for height_m in np.arange(0.,2.51,.5):
                    row = int(round((height-1)/2-(height/2)/1.25*(height_m-1.25)))
                    row = min(height-1,max(0,row))
                    draw.line([(0,row),(width-1,row)],fill=(60,220,220),width=1)
                    draw.text((5,min(height-13,max(1,row-12))),f'{height_m:.1f} m',fill=(255,255,255),stroke_width=1,stroke_fill=(0,0,0))
                qc.save(output/f'qc-metric-{i:06d}.png')
            if i%100 == 0:
                print(json.dumps(dict(rendered=i+1,total=len(poses))), flush=True)
    finally:
        for encoder in encoders:
            encoder.close()
        if not reuse:
            viewer.close()
    np.savez_compressed(sensors/'frames.npz', frame_index=np.arange(len(ticks)), physics_step=ticks*4,
        timestamp=ticks/50, scene_from_optical=np.asarray(camera_poses), depth_sha256=np.asarray(depth_hashes))
    if np.any(model.cam_sensorsize[0] != 0):
        raise ValueError('This calibrated render profile requires the fovy camera model')
    focal = height/(2*math.tan(math.radians(float(model.cam_fovy[0]))/2))
    camera = dict(width=width,height=height,fps=25,intrinsics=[[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1]],
        depth_scale=.001,depth_min_m=.001,depth_max_m=65.535,invalid_value=0,depth_definition='optical_z',
        optical_convention='right-handed x-right y-down z-forward',body_name=meta['body_names'][viewer.ego_body],
        mount_offset=viewer.ego_offset,mount_forward=viewer.ego_forward,up_policy='world_z',
        rgb_depth_aligned=True,depth_codec='ffv1',depth_codec_level=3,depth_pixel_format='gray16le')
    manifest['versions']['render']=dict(runtime=runtime(),code=repository(ROOT),
        asset_manifest=file_identity(visual/'asset.json'),visual_model=file_identity(visual/meta['entry']))
    manifest['camera']=camera
    manifest['embodiment']['camera']=camera
    manifest['recording_complete']=True
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'render_report.json').write_text(json.dumps(dict(frames=len(ticks),rgb_depth_same_dispatch=True,
        backend='mujoco_warp',source='actual_physics_body_poses',width=width,height=height,
        replay_fps=25/replay_stride,replay_capture_stride=replay_stride,
        qc_vertical_framing_m=2.5,qc_grid_step_m=.5,qc_grid_plane='vertical plane through robot root',qc_ground_z_m=0.),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--execution', type=Path)
    inputs.add_argument('--batch', type=Path)
    parser.add_argument('--replay-stride', type=int, choices=(1,5), default=1)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()
    if args.batch:
        try:
            for execution in sorted(args.batch.glob('env-*')):
                manifest = json.loads((execution/'manifest.json').read_text())
                if manifest['status'] == 'complete' and not manifest.get('execution_error'):
                    render(execution, args.width, args.height, reuse=True, replay_stride=args.replay_stride)
        finally:
            for _, viewer in _RENDER_CACHE.values():
                viewer.close()
    else:
        render(args.execution, args.width, args.height, replay_stride=args.replay_stride)
