"""Equipment mounts from batched device transforms, including shield clearance."""
import torch

from .hand_equipment import SHIELD_HANDLE


class TensorHandMounts:
    def __init__(self, source, device, dtype=torch.float64):
        self.device, self.dtype = torch.device(device), dtype
        tensor = lambda value: torch.as_tensor(value, dtype=dtype, device=self.device)
        self.hands = {side: (index, tensor(frame), tensor(center))
                      for side, (index, frame, center) in source.hands.items()}
        self.forearm = source.forearm
        self.bind = tensor(source.palette.bind) if source.palette is not None else None
        self.handle = tensor(SHIELD_HANDLE)
        self.shield_rotation = tensor([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
        self.angles = torch.deg2rad(torch.arange(-180., 181., device=self.device, dtype=dtype))
        self.cos, self.sin = self.angles.cos(), self.angles.sin()

    def mounts(self, matrices):
        if matrices.device != self.device or matrices.dtype != self.dtype:
            raise ValueError('Mount palettes must match prepared device and dtype')
        result = {}
        for side, (index, frame, center) in self.hands.items():
            rotation, origin = matrices[:, index, :3, :3], matrices[:, index, :3, 3]
            rot = rotation @ frame
            grip = origin + rotation @ center
            if side == 'l':
                rot = rot @ self.shield_rotation
                if self.forearm is not None:
                    elbow = (matrices[:, self.forearm, :3, 3] if self.bind is None else
                             (matrices[:, self.forearm] @ self.bind[self.forearm])[:, :3, 3])
                    across = rot[:, :, 0]
                    proximal = elbow-grip
                    proximal = proximal-across*(across*proximal).sum(-1, keepdim=True)
                    length = torch.linalg.vector_norm(proximal, dim=-1, keepdim=True)
                    proximal = proximal/length.clamp_min(1e-30)
                    normal = torch.linalg.cross(proximal, across)
                    wrist = (matrices[:, index, :3, 3] if self.bind is None else
                             (matrices[:, index] @ self.bind[index])[:, :3, 3])
                    offsets = torch.stack((elbow-grip, wrist-grip), dim=1)
                    limit = .07-.055
                    needs_search = (offsets*normal[:, None]).sum(-1).amax(-1) > limit
                    candidates = self.cos[None, :, None]*normal[:, None]+self.sin[None, :, None]*proximal[:, None]
                    valid = (offsets @ candidates.transpose(1, 2)).amax(1) <= limit
                    choice = torch.where(valid, self.angles.abs(), torch.inf).argmin(-1)
                    rows = torch.arange(len(matrices), device=self.device)
                    use = needs_search & valid[rows, choice]
                    normal = torch.where(use[:, None], candidates[rows, choice], normal)
                    proximal = torch.where(use[:, None], torch.linalg.cross(across, normal), proximal)
                    aligned = torch.stack((across, normal, proximal), dim=-1)
                    rot = torch.where((length > 1e-6)[:, :, None], aligned, rot)
            mount = torch.eye(4, device=self.device, dtype=self.dtype).expand(len(matrices), 4, 4).clone()
            mount[:, :3, :3] = rot
            mount[:, :3, 3] = grip-(rot @ self.handle if side == 'l' else 0.)
            result['shield' if side == 'l' else 'right_hand'] = mount
        return result
