import torch
from torch import Tensor


class ActionTransform:
    """Normalized action -> PD position target, as declared by the package."""

    def __init__(self, spec: dict, device: torch.device):
        self.transform = spec["transform"]
        self.clamp = float(spec.get("clamp") or 1.0)
        self.offset = torch.tensor(spec["offset"], dtype=torch.float32, device=device)
        self.scale = torch.tensor(spec["scale"], dtype=torch.float32, device=device)

    def __call__(self, action: Tensor) -> Tensor:
        if self.transform == "clamp":
            action = action.clamp(-self.clamp, self.clamp)
        elif self.transform == "tanh":
            action = action.tanh()
        return self.offset + self.scale * action
