"""Native G1/A3 poses and prepared equipment transforms on the render device."""
import torch
import warp as wp

from .hand_equipment import HandMounts
from .hand_mounts_tensor import TensorHandMounts
from .soma_palette_tensor import rotations_xyzw


@wp.kernel
def _append_mounts(matrices: wp.array[wp.mat44], offset: int, poses: wp.array[wp.transform]):
    i = wp.tid()
    m = matrices[i]
    r = wp.mat33(m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1], m[2, 2])
    poses[offset+i] = wp.transform(wp.vec3(m[0, 3], m[1, 3], m[2, 3]), wp.quat_from_matrix(r))


class NativeDevicePoses:
    def __init__(self, meta, device):
        self.device = torch.device(str(device))
        self.stream = wp.stream_to_torch(str(device))
        self.count = meta['num_agents']
        self.bodies = len(meta['policy_body_names'])
        self.size = self.count*self.bodies
        self.mounts = None
        with torch.cuda.stream(self.stream):
            if meta.get('armed'):
                source = HandMounts(meta['policy_body_names'], self.count, 'cpu', centers=meta['armed']['centers'], sockets=meta['armed'].get('sockets'))
                self.mounts = TensorHandMounts(source, self.device)
        self.output = wp.empty(self.size+(2*self.count if self.mounts else 0), dtype=wp.transform, device=str(device))
        self.keepalive = None

    def update(self, poses):
        if poses.device != self.device or poses.shape != (self.size, 7) or poses.dtype != torch.float32:
            raise ValueError('Native render poses require matching CUDA float32 body layout')
        self.stream.wait_stream(torch.cuda.current_stream(poses.device))
        with torch.cuda.stream(self.stream), wp.ScopedDevice(str(self.device)):
            torch._assert_async(torch.isfinite(poses).all(), 'Nonfinite native poses')
            torch._assert_async((torch.abs(torch.linalg.vector_norm(poses[:, 3:], dim=-1)-1) <= 1e-3).all(),
                                'Native rotations must be unit quaternions')
            physical = poses.contiguous()
            wp.copy(self.output, wp.from_torch(physical, dtype=wp.transform), count=self.size)
            mounts = None
            if self.mounts:
                q = poses.reshape(self.count, self.bodies, 7).double()
                matrices = torch.eye(4, device=self.device, dtype=torch.float64).expand(self.count, self.bodies, 4, 4).clone()
                matrices[:, :, :3, :3] = rotations_xyzw(q[:, :, 3:])
                matrices[:, :, :3, 3] = q[:, :, :3]
                result = self.mounts.mounts(matrices)
                mounts = torch.stack((result['shield'], result['right_hand']), dim=1).reshape(-1, 4, 4).float().contiguous()
                wp.launch(_append_mounts, dim=2*self.count,
                          inputs=[wp.from_torch(mounts, dtype=wp.mat44), self.size], outputs=[self.output])
            poses.record_stream(self.stream)
            self.keepalive = physical, mounts
        return self.output
