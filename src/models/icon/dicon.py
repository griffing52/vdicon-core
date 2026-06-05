import torch
import torch.nn as nn


class FlowMatchingExpert(nn.Module):
    """Conditional velocity field used by D-ICON flow matching."""

    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, but got {num_layers}")

        layers: list[nn.Module] = []
        in_dim = 2 * state_dim + 1
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity from noised state, deterministic condition, and time."""
        t_feature = torch.broadcast_to(t, x_t.shape[:-1] + (1,))
        model_input = torch.cat([x_t, cond, t_feature], dim=-1)
        return self.net(model_input)


class DICON(nn.Module):
    """Diffusion-augmented ICON with a conditional flow-matching expert."""

    def __init__(self, backbone: nn.Module, flow_expert: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.flow_expert = flow_expert

    def forward(self, data: dict[str, torch.Tensor], mode: str, **kwargs) -> torch.Tensor:
        """Keep ICON-compatible forward API by delegating to the deterministic backbone."""
        if mode == "train":
            return self.backbone(data, mode=mode)
        return self.backbone(data, mode=mode, **kwargs)

    def predict_backbone(self, data: dict[str, torch.Tensor], mode: str, need_weights: bool) -> torch.Tensor | tuple:
        """Run deterministic ICON prediction."""
        if mode == "train":
            return self.backbone(data, mode=mode)
        return self.backbone(data, mode=mode, need_weights=need_weights)

    def compute_flow_matching_loss(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Compute per-sample conditional flow-matching loss."""
        batch_size = target.shape[0]
        dtype = target.dtype
        device = target.device

        t = torch.rand(batch_size, device=device, dtype=dtype)
        t_view = t.view(batch_size, *([1] * (target.ndim - 1)))
        noise = torch.randn_like(target)

        x_t = (1.0 - t_view) * noise + t_view * target
        velocity_target = target - noise
        velocity_pred = self.flow_expert(x_t=x_t, cond=cond, t=t_view)

        return (velocity_pred - velocity_target).pow(2).flatten(1).mean(dim=1)

    def sample_from_flow(self, cond: torch.Tensor, num_steps: int, num_samples: int) -> torch.Tensor:
        """Generate predictions by integrating the learned flow field with Euler steps."""
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, but got {num_steps}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, but got {num_samples}")

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
                t_view = t.view(batch_size, *([1] * (cond.ndim - 1)))
                velocity = self.flow_expert(x_t=x, cond=cond, t=t_view)
                x = x + dt * velocity
            sample_list.append(x)

        return torch.stack(sample_list, dim=0).mean(dim=0)
