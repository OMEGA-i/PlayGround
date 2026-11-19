"""MJWarp pose viewer implementing the Web renderer process interface, without Newton."""
import gc
import io
import math
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from omega_playground.policy.package import load
from omega_playground.viewer.mjwarp import MjWarpPoseRenderer
from omega_playground.viewer.gpu_pixels import Rgb8Readback
from omega_playground.viewer.mjwarp_asset import load_visual_model


def mjwarp_viewer(package_path, visual_path, scene_sha256, count, device, *, armed=False,
                  skin_manifest_sha256=None, **kwargs):
    environment_path = kwargs.pop('environment_path', None)
    environment = None
    if environment_path:
        from .environment_asset import load as load_environment
        environment = load_environment(environment_path, scene_sha256)
    package = load(package_path)
    if skin_manifest_sha256 is not None:
        from omega_playground.viewer.mjwarp_skin_asset import load_prepared_skin
        model, painter, bridge, meta = load_prepared_skin(visual_path, package, agents=count,
            armed=armed, device=device, scene_sha256=scene_sha256,
            skin_manifest_sha256=skin_manifest_sha256)
        return MjWarpViewer(model, meta, visual_directory=visual_path, device=device, skin=(painter, bridge), environment=environment, **kwargs)
    model, meta = load_visual_model(visual_path, package, scene_sha256, count, armed=armed)
    return MjWarpViewer(model, meta, visual_directory=visual_path, device=device, environment=environment, **kwargs)


class MjWarpViewer:
    freeze_static_gc = True

    def __init__(self, model, meta, *, visual_directory=None, device='cuda:0', width=640, height=480, video=None,
                 fps=Fraction(25), video_codec='h265', video_bitrate_kbps=1500, camera='qc', profile=False,
                 skin=None, pbr=False, linear_output=False, shadows=True, textures=True, environment=None):
        if video_codec not in ('h264', 'h265') or not 128 <= video_bitrate_kbps <= 50000:
            raise ValueError('Invalid video encoding settings')
        supersample = int(os.environ.get("OMEGA_RENDER_SUPERSAMPLE", "1"))
        if supersample not in (1, 2, 3, 4):
            raise ValueError("OMEGA_RENDER_SUPERSAMPLE must be 1, 2, 3 or 4")
        self.edge_antialias = os.environ.get("OMEGA_RENDER_EDGE_AA") == "1"
        self.supersample = supersample
        self.renderer = MjWarpPoseRenderer(model, meta['body_names'], device=device, width=width*supersample, height=height*supersample,
                                          static_instances=meta.get('instance_transforms', ()),
                                          backface_culling=skin is None, shadows=shadows, pbr=pbr, linear_output=linear_output,
                                          texture_color_spaces=meta.get('texture_color_spaces', {}),
                                          texture_wrap_modes=meta.get('texture_wrap_modes', {}), textures=textures,
                                          skin_meshes=skin is not None, cutout_materials=meta.get('cutout_materials'),
                                          visual_directory=visual_directory, face_varying_normals=meta.get('face_varying_normals'))
        self.environment_meta = None
        if environment is not None:
            arrays, self.environment_meta = environment
            self.renderer.set_prepared_environment(arrays)
        self.rgb_readback = Rgb8Readback((1, height, width), device, supersample=supersample, edge_antialias=self.edge_antialias)
        self.skin = skin
        if skin is not None:
            skin[1].attach(self.renderer)
        self.armed_adapter = None
        if meta.get('armed') and skin is None:
            from omega_playground.viewer.mjwarp_armed import ArmedPoseAdapter
            self.armed_adapter = ArmedPoseAdapter(meta)
        self.ego_jpeg = None
        names = meta['policy_body_names'] if skin is not None else meta['body_names']
        # Offsets and forward axes are in the physical head link's local frame.
        self.ego_body=None
        for link,offset,forward in [('head_pitch_Link',[.15,0.,.06],[1.,0.,0.]),
                                    ('head_yaw_link',[.15,0.,.06],[1.,0.,0.]),
                                    ('Head',[0.,-.15,.07],[0.,-1.,0.]),
                                    ('torso_link',[.14,0.,.43],[1.,0.,0.])]:
            if link in names:
                self.ego_body,self.ego_offset,self.ego_forward=names.index(link),offset,forward
                break
        self.meta, self.width, self.height = meta, width, height
        self.video = Path(video) if video else None
        self.fps, self.video_codec, self.video_bitrate_kbps = fps, video_codec, video_bitrate_kbps
        self.stream_video, self.frame_sink = True, None
        self.encoder = self.latest_jpeg = None
        self.frames, self.render_times = 0, []
        self.follow_body, self.root_bodies = 0, [0]
        self.camera_mode, self.camera_ground_z = camera, 0.
        self.orbit_yaw, self.orbit_pitch, self.orbit_distance = 30., -12., 4.
        self.fovy = float(model.cam_fovy[0])
        self.formation_camera = None
        self.camera_model = model
        self.gc_events, self.render_stages = [], []
        self.gc_start = {}
        self.profile = profile
        if profile:
            gc.callbacks.append(self._record_gc)

    def _record_gc(self, phase, info):
        now = time.perf_counter()
        generation = info['generation']
        if phase == 'start':
            self.gc_start[generation] = now
        else:
            self.gc_events.append({'generation': generation, 'wall': now,
                                   'ms': (now-self.gc_start.get(generation, now))*1000})

    def show(self, body_q, sim_time, record=True):
        start = time.perf_counter()
        device_input = getattr(body_q, 'is_cuda', False)
        def camera_values(value):
            if device_input:
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=float)
        root = camera_values(body_q[self.follow_body, :3])
        target = root.copy()
        if self.camera_mode in ('squad', 'top'):
            roots = camera_values(body_q[self.root_bodies, :3])
            target = roots.mean(axis=0)
            radius = max(2., np.linalg.norm(roots[:, :2]-target[:2], axis=1).max()+1.5)
            distance = radius/math.tan(math.radians(self.fovy)/2)
            if self.camera_mode == 'top':
                eye = target+np.array([.01, 0, distance])
            else:
                if self.formation_camera is None:
                    from omega_playground.viewer.formation_camera import FormationCamera, scene_bounds
                    self.formation_camera = FormationCamera(scene_bounds(
                        self.camera_model, self.meta.get('instance_transforms', ())))
                eye = self.formation_camera.eye(roots, target, distance)
        elif self.camera_mode == 'qc':
            target[2] = (self.camera_ground_z or 0.)+1.
            distance = 1.25/math.tan(math.radians(self.fovy)/2)
            eye = target+[-.866*distance, -.5*distance, 0.]
        else:
            yaw, pitch = math.radians(self.orbit_yaw), math.radians(self.orbit_pitch)
            distance = self.orbit_distance if self.camera_mode == 'free' else 4.
            eye = target-distance*np.array([math.cos(pitch)*math.cos(yaw), math.cos(pitch)*math.sin(yaw), math.sin(pitch)])
        self.renderer.look_at(eye, target)
        camera_done = time.perf_counter()
        if self.skin is not None:
            painter, bridge = self.skin
            painter.log(bridge, body_q)
            bridge.refit()
            render_poses = bridge.poses()
        else:
            if device_input:
                if not hasattr(self, '_native_device_poses'):
                    from .native_poses_tensor import NativeDevicePoses
                    self._native_device_poses = NativeDevicePoses(self.meta, body_q.device)
                render_poses = self._native_device_poses.update(body_q)
            else:
                render_poses = self.armed_adapter.extend(body_q) if self.armed_adapter else body_q
        frame = self.renderer.render(render_poses)
        dispatch_done = time.perf_counter()
        rgb = self.rgb_readback.read(frame)[0].copy()
        if self.ego_body is not None and self.frame_sink:
            # G1's head mesh is rigidly attached to its torso.
            from scipy.spatial.transform import Rotation
            pose = camera_values(body_q[self.ego_body])
            orientation = Rotation.from_quat(pose[3:7])
            eye = pose[:3] + orientation.apply(self.ego_offset)
            self.renderer.look_at(eye, eye + orientation.apply(self.ego_forward))
            ego = self.rgb_readback.read(self.renderer.render(render_poses))[0]
            buffer = io.BytesIO()
            Image.fromarray(ego).save(buffer, format='JPEG', quality=85)
            self.ego_jpeg = buffer.getvalue()
        if device_input:
            # The frame acknowledgement lets the producer reuse its IPC slot.
            import torch
            torch.cuda.synchronize(body_q.device)
        readback_done = time.perf_counter()
        if self.profile and record:
            self.render_stages.append({'camera_ms': (camera_done-start)*1000,
                                       'dispatch_ms': (dispatch_done-camera_done)*1000,
                                       'readback_ms': (readback_done-dispatch_done)*1000,
                                       'pixels_ms': (time.perf_counter()-readback_done)*1000})
        if record:
            self.render_times.append((time.perf_counter()-start)*1000)
        if self.frame_sink:
            self.frame_sink(rgb, record)
        else:
            self.encode_frame(rgb, record)
        self.frames += int(record)

    def encode_frame(self, rgb, record):
        preview = io.BytesIO()
        Image.fromarray(rgb).save(preview, format='JPEG', quality=85)
        self.latest_jpeg = preview.getvalue()
        if record and self.video:
            if self.encoder is None:
                self.video.parent.mkdir(parents=True, exist_ok=True)
                codec = (['-c:v', 'hevc_nvenc', '-preset', 'p1', '-tune', 'ull', '-bf', '0',
                          '-rc', 'cbr', '-b:v', f'{self.video_bitrate_kbps}k',
                          '-maxrate', f'{self.video_bitrate_kbps}k', '-bufsize', f'{self.video_bitrate_kbps//2}k',
                          '-profile:v', 'main', '-level:v', '4.1', '-tag:v', 'hvc1', '-forced-idr', '1']
                         if self.video_codec == 'h265' else
                         ['-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2', '-tune', 'zerolatency',
                          '-profile:v', 'baseline', '-level:v', '3.1', '-sc_threshold', '0'])
                self.encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', f'{self.width}x{self.height}', '-r', str(self.fps), '-i', 'pipe:0', '-an', *codec,
                    '-g', str(max(1, round(float(self.fps)/5))), '-pix_fmt', 'yuv420p',
                    '-movflags', 'frag_keyframe+empty_moov+default_base_moof', '-flush_packets', '1', str(self.video)],
                    stdin=subprocess.PIPE)
            self.encoder.stdin.write(rgb.tobytes())
        return self.latest_jpeg

    def close(self):
        if self._record_gc in gc.callbacks:
            gc.callbacks.remove(self._record_gc)
        if self.encoder:
            self.encoder.stdin.close()
            if self.encoder.wait() != 0:
                raise RuntimeError('MJWarp video encoder failed')
            self.encoder = None
        updater = getattr(self.skin[0], '_tensor_updater', None) if self.skin else None
        skin_runtime = None if not self.skin else {
            'implementation': type(updater).__name__ if updater is not None else 'host_palette',
            'cuda_graph': updater.graph is not None if updater is not None else False,
        }
        return {'backend': 'mujoco_warp', 'python': sys.executable, 'frames': self.frames, 'fps': str(self.fps),
                'skin_runtime': skin_runtime,
                'video': str(self.video) if self.video else None, 'render_and_readback_ms': self.render_times,
                'frozen_gc_objects': gc.get_freeze_count(), 'render_stages': self.render_stages, 'gc_events': self.gc_events,
                'newton_imported': 'newton' in sys.modules, 'supersample': self.supersample, 'edge_antialias': self.edge_antialias,
                'output_resolution': [self.width, self.height],
                'internal_resolution': [self.width*self.supersample, self.height*self.supersample], 'environment_asset': self.environment_meta, 'visual_gaps': self.meta.get('visual_gaps', [])}
