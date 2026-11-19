"""Batched SOMA display palettes on the input device; no physics or host readback.

The authored hierarchy/calibration is prepared by Soma23Palette once. This
adapter keeps its sole-support semantics; it does not retarget physical joints.
"""
import torch


def rotations_xyzw(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


class TensorSoma23Palette:
    def __init__(self, source, device, dtype=torch.float64):
        self.device, self.dtype = torch.device(device), dtype
        tensor = lambda x: torch.as_tensor(x, device=self.device, dtype=dtype)
        self.bind, self.inverse = tensor(source.bind), tensor(source.inverse)
        self.calibration, self.root_offset = tensor(source.calibration), tensor(source.root_offset)
        self.parents = tuple(int(p) for p in source.parents)
        self.root = source.root
        # Authored bind relations never change with a pose. Prepare once rather
        # than recording a second matrix multiply for every bone in every frame.
        self.relative_bind = tuple(tensor(source.inverse[parent] @ source.bind[i])
                                   if parent >= 0 else None for i, parent in enumerate(self.parents))
        self.root_offsets = tuple(tensor(source.bind[i, :3, 3]-source.bind[self.root, :3, 3]
                                         + source.root_offset) if parent < 0 else None
                                  for i, parent in enumerate(self.parents))
        self.hips = source.soma_names.index('Hips')
        self.hips_body = int(source.indices[self.hips])
        self.indices = torch.tensor(source.indices, device=self.device)
        self.mapping = tuple(source.soma_names.index(source.mapping[n]) if n in source.mapping else -1
                             for n in source.names)
        self.sole = None
        if source.sole is not None:
            s = source.sole
            self.sole = (torch.tensor(s.joints, device=self.device), tensor(s.weighted_points),
                         torch.tensor(s.owners, device=self.device), tensor(s.corners))

    def matrices(self, poses):
        if poses.device != self.device or poses.dtype != self.dtype:
            raise ValueError('Palette poses must match the prepared device and dtype')
        if poses.ndim != 3 or poses.shape[-1] != 7:
            raise ValueError('Expected agent-major [agents, bodies, XYZ+XYZW]')
        positions, rotations = poses[..., :3], rotations_xyzw(poses[..., 3:])
        ordered = rotations[:, self.indices]
        root_pos = positions[:, self.hips_body]
        root_rot = ordered[:, self.hips]
        targets = []
        for i, parent in enumerate(self.parents):
            if parent >= 0:
                target = targets[parent] @ self.relative_bind[i]
            else:
                target = torch.eye(4, dtype=self.dtype, device=self.device).expand(len(poses), 4, 4).clone()
                target[:, :3, :3] = root_rot @ self.bind[i, :3, :3]
                target[:, :3, 3] = root_pos + (root_rot @ self.root_offsets[i])
            if self.mapping[i] >= 0:
                target[:, :3, :3] = ordered[:, self.mapping[i]] @ self.calibration[i]
            targets.append(target)
        matrices = torch.stack(targets, dim=1) @ self.inverse
        if self.sole is not None:
            joints, weighted, owners, corners = self.sole
            visual = (matrices[:, joints, 2, :] * weighted).sum((-1, -2)).amin(-1)
            physical = ((rotations[:, owners, 2, :] * corners).sum(-1)
                        + positions[:, owners, 2]).amin(-1)
            matrices[:, :, 2, 3] += (physical-visual)[:, None]
        return matrices
