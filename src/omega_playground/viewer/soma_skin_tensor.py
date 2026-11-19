"""Live SOMA skin/equipment update on one shared Torch/Warp CUDA stream."""
import os

import numpy as np
import torch
import warp as wp

from .soma_palette_tensor import TensorSoma23Palette
from .hand_mounts_tensor import TensorHandMounts
from .skin import _skin


@wp.kernel
def _rigid_poses(matrices: wp.array[wp.mat44], ids: wp.array[int], poses: wp.array[wp.transform]):
    i = wp.tid()
    m = matrices[i]
    rotation = wp.mat33(m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1], m[2, 2])
    poses[ids[i]] = wp.transform(wp.vec3(m[0, 3], m[1, 3], m[2, 3]), wp.quat_from_matrix(rotation))


class TensorSkinUpdater:
    def __init__(self, painter, bridge):
        self.painter, self.bridge = painter, bridge
        self.device = painter.device
        self.stream = wp.stream_to_torch(self.device)
        with torch.cuda.stream(self.stream):
            self.palette = TensorSoma23Palette(painter.adapter, self.device)
            self.mounts = TensorHandMounts(painter.hand_equipment, self.device) if painter.hand_equipment else None
        # Topology and identity poses are established once by the prepared loader.
        self.poses = wp.array(bridge.poses(), dtype=wp.transform, device=self.device)
        names = list(bridge.meshes)
        self.entries = []
        for agent in range(painter.num_agents):
            if self.mounts:
                for part in painter.hand_equipment.parts:
                    self.entries.append((names.index(f'hand_prop_{agent:03d}_{part["name"]}'), agent, part['body']))
            if painter.rigid_skin:
                for slot, bone in enumerate(painter.rigid_bones):
                    self.entries.append((names.index(f'skin_{agent:03d}_{slot}'), agent, bone))
        self.ids = wp.array([entry[0] for entry in self.entries], dtype=wp.int32, device=self.device)
        bridge.device_poses = self.poses
        self.keepalive = None
        self.graph = None
        self.capture = os.environ.get('OMEGA_SOMA_GPU_SKIN_GRAPH', '1') != '0'
        with torch.cuda.stream(self.stream):
            self.input = torch.empty((painter.num_agents, painter.bodies, 7),
                                     dtype=torch.float64, device=self.device)

    def _compute(self):
        matrices = self.palette.matrices(self.input)
        mounts = self.mounts.mounts(matrices) if self.mounts else {}
        rigid = None
        if self.entries:
            rigid = torch.stack([mounts[entry][agent] if isinstance(entry, str) else matrices[agent, entry]
                                 for _, agent, entry in self.entries]).float().contiguous()
        return matrices.float().contiguous(), rigid

    def _capture(self):
        # Initialization only: warm/capture on a non-default stream, then consume
        # persistent outputs on the shared Warp stream. Never capture host copies.
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(self.stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                self._compute()
        capture_stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=capture_stream):
            self.keepalive = self._compute()
        self.stream.wait_stream(capture_stream)

    def _copy_input(self, body_q):
        """Own the pose snapshot on the render stream without a device readback.

        CUDA callers publish on their current stream (IPC callers must first
        wait for the producer). Copying into our persistent buffer preserves
        graph addresses and releases the source once stream work completes.
        """
        p = self.painter
        shape = (p.num_agents*p.bodies, 7)
        if isinstance(body_q, torch.Tensor):
            if body_q.shape != shape or body_q.device != self.input.device:
                raise ValueError('Expected authoritative poses on the render CUDA device')
            if body_q.dtype not in (torch.float32, torch.float64):
                raise ValueError('Expected float32 or float64 authoritative poses')
            self.stream.wait_stream(torch.cuda.current_stream(body_q.device))
            with torch.cuda.stream(self.stream):
                torch._assert_async(torch.isfinite(body_q).all(), 'Nonfinite authoritative body poses')
                self.input.copy_(body_q.reshape(self.input.shape))
                body_q.record_stream(self.stream)
        else:
            values = np.asarray(body_q)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError('Expected finite authoritative body poses')
            with torch.cuda.stream(self.stream):
                self.input.copy_(torch.as_tensor(values.reshape(self.input.shape)))

    def log(self, body_q):
        p = self.painter
        self._copy_input(body_q)
        with torch.cuda.stream(self.stream), wp.ScopedDevice(self.device):
            if self.capture:
                if self.graph is None:
                    self._capture()
                self.graph.replay()
            else:
                self.keepalive = self._compute()
            palettes, rigid = self.keepalive
            if self.entries:
                wp.launch(_rigid_poses, dim=len(self.entries),
                          inputs=[wp.from_torch(rigid, dtype=wp.mat44), self.ids], outputs=[self.poses])
            if not p.rigid_skin:
                for agent in range(p.num_agents):
                    palette = wp.from_torch(palettes[agent], dtype=wp.mat44)
                    for part in p.parts:
                        points, normals = part['outputs'][agent]
                        wp.launch(_skin, dim=len(part['points']),
                                  inputs=[part['points'], part['normals'], part['joints'], part['weights'],
                                          palette, part['weights'].shape[1]], outputs=[points, normals])
                self.bridge.update_deformed_meshes(p.batch_slices, p.batch_points, p.batch_normals)
            # Retain shared Torch allocations until subsequent same-stream work.
            # self.keepalive owns the buffers used by Warp and CUDA Graph replay.
