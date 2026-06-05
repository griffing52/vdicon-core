from __future__ import annotations

import torch
import torch.nn as nn


class ImageFlowMatchingExpert(nn.Module):
    """Convolutional conditional velocity field for 2D flow matching."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, but got {num_layers}")

        layers: list[nn.Module] = []
        current_channels = 2 * in_channels + 1
        for layer_idx in range(num_layers - 1):
            next_channels = hidden_channels
            layers.append(nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(num_groups=max(1, min(8, next_channels)), num_channels=next_channels))
            layers.append(nn.SiLU())
            current_channels = next_channels
        layers.append(nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None, None, None]
        elif t.ndim == 2:
            t = t[:, :, None, None]
        t_feature = torch.broadcast_to(t, x_t.shape[:1] + (1,) + x_t.shape[2:])
        model_input = torch.cat([x_t, cond, t_feature], dim=1)
        return self.net(model_input)


class ViconDICON(nn.Module):
    """VICON backbone with a 2D flow-matching refinement head."""

    def __init__(self, backbone: nn.Module, flow_expert: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.flow_expert = flow_expert

    @staticmethod
    def _squeeze_sample_dim(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 5 and tensor.shape[1] == 1:
            return tensor[:, 0]
        return tensor

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(f, g)

    def predict_backbone(self, f: torch.Tensor, g: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(f, g)

    def compute_flow_matching_loss(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        target = self._squeeze_sample_dim(target)
        cond = self._squeeze_sample_dim(cond)

        if target.ndim != 4 or cond.ndim != 4:
            raise ValueError("target and cond must be 4D tensors shaped [B, C, H, W]")

        batch_size = target.shape[0]
        dtype = target.dtype
        device = target.device

        t = torch.rand(batch_size, device=device, dtype=dtype)
        t_view = t.view(batch_size, 1, 1, 1)
        noise = torch.randn_like(target)

        x_t = (1.0 - t_view) * noise + t_view * target
        velocity_target = target - noise
        velocity_pred = self.flow_expert(x_t=x_t, cond=cond, t=t_view)

        return (velocity_pred - velocity_target).pow(2).flatten(1).mean(dim=1)

    def sample_from_flow(self, cond: torch.Tensor, num_steps: int, num_samples: int) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, but got {num_steps}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, but got {num_samples}")

        cond = self._squeeze_sample_dim(cond)
        batch_size = cond.shape[0]
        dtype = cond.dtype
        device = cond.device
        dt = 1.0 / num_steps
        sample_list = []

        for _ in range(num_samples):
            x = torch.randn_like(cond)
            for step in range(num_steps):
                t_value = (step + 0.5) / num_steps
                t = torch.full((batch_size,), t_value, device=device, dtype=dtype)
                t_view = t.view(batch_size, 1, 1, 1)
                velocity = self.flow_expert(x_t=x, cond=cond, t=t_view)
                x = x + dt * velocity
            sample_list.append(x)

        return torch.stack(sample_list, dim=0).mean(dim=0)