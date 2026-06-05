from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm_channels(num_channels: int) -> int:
    return max(1, min(8, num_channels))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, but got {embedding_dim}")
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        elif t.ndim > 2:
            t = t.view(t.shape[0], -1)

        half_dim = self.embedding_dim // 2
        device = t.device
        dtype = t.dtype
        freq = torch.arange(half_dim, device=device, dtype=dtype)
        freq = torch.exp(-math.log(10000.0) * freq / max(half_dim - 1, 1))
        args = t * freq[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_channels(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_norm_channels(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_channels, 2 * out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(time_embedding).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h) * (1.0 + scale) + shift))
        return h + self.skip(x)


class ResidualLatentFlowMatchingExpert(nn.Module):
    """A stronger conditional flow head with a small U-Net and time embeddings."""

    def __init__(self, target_channels: int, cond_channels: int, base_channels: int, time_channels: int) -> None:
        super().__init__()
        self.target_channels = target_channels
        self.cond_proj = nn.Sequential(
            nn.Conv2d(cond_channels, base_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.time_embed = SinusoidalTimeEmbedding(time_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )

        self.stem = nn.Conv2d(target_channels + base_channels, base_channels, kernel_size=3, padding=1)
        self.block1 = ResBlock(base_channels, base_channels, time_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.block2 = ResBlock(base_channels * 2, base_channels * 2, time_channels)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_channels)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_channels)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.block3 = ResBlock(base_channels * 4, base_channels * 2, time_channels)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.block4 = ResBlock(base_channels * 2, base_channels, time_channels)
        self.out_norm = nn.GroupNorm(_group_norm_channels(base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, target_channels, kernel_size=3, padding=1)

    def _prepare_time(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            t = t.view(t.shape[0])
        time_embedding = self.time_embed(t)
        return self.time_mlp(time_embedding)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim == 5 and x_t.shape[1] == 1:
            x_t = x_t[:, 0]
        if cond.ndim == 5 and cond.shape[1] == 1:
            cond = cond[:, 0]
        if cond.shape[-2:] != x_t.shape[-2:]:
            cond = F.interpolate(cond, size=x_t.shape[-2:], mode="bilinear", align_corners=False)

        cond = self.cond_proj(cond)
        time_embedding = self._prepare_time(t)

        h0 = self.stem(torch.cat([x_t, cond], dim=1))
        h1 = self.block1(h0, time_embedding)
        h2 = self.block2(self.down1(h1), time_embedding)
        h3 = self.mid1(self.down2(h2), time_embedding)
        h3 = self.mid2(h3, time_embedding)
        h = self.up2(h3)
        h = self.block3(torch.cat([h, h2], dim=1), time_embedding)
        h = self.up1(h)
        h = self.block4(torch.cat([h, h1], dim=1), time_embedding)
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


class ViconLatentDICONV2(nn.Module):
    """Residual latent DICON with cosine flow bridge and Heun sampling."""

    def __init__(
        self,
        backbone: nn.Module,
        flow_expert: nn.Module,
        loss_time_weight: float = 1.0,
        loss_time_power: float = 2.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.flow_expert = flow_expert
        self.loss_time_weight = loss_time_weight
        self.loss_time_power = loss_time_power

    @staticmethod
    def _prompt_normalization(x: torch.Tensor):
        mean = x.mean(dim=(1, 3, 4), keepdim=True)
        std = x.std(dim=(1, 3, 4), keepdim=True) + 1e-5
        x_normalized = (x - mean) / std
        return x_normalized, mean, std

    def _build_prompt(self, data: dict[str, torch.Tensor]):
        dummy_label = torch.zeros_like(data["ex_g"][:, -1:, :, :, :])
        g = data["ex_g"]
        f = torch.cat((data["ex_f"], data["qn_f"]), dim=1)
        f_norm, _, _ = self._prompt_normalization(f)
        g_norm, g_mean, g_std = self._prompt_normalization(g)
        g_norm = torch.cat((g_norm, dummy_label), dim=1)
        return f_norm, g_norm, g_mean, g_std

    def _run_backbone(self, data: dict[str, torch.Tensor], mode: str, need_weights: bool):
        if hasattr(self.backbone, "predict_backbone"):
            return self.backbone.predict_backbone(data=data, mode=mode, need_weights=need_weights)
        if need_weights:
            return self.backbone(data["ex_f"], data["ex_g"])
        return self.backbone(data["ex_f"], data["ex_g"])

    def forward(self, data: dict[str, torch.Tensor], mode: str, **kwargs):
        return self.predict_backbone(data=data, mode=mode, need_weights=kwargs.get("need_weights", False))

    def predict_backbone(self, data: dict[str, torch.Tensor], mode: str, need_weights: bool):
        f_norm, g_norm, g_mean, g_std = self._build_prompt(data)
        outputs = self._run_backbone({"ex_f": f_norm, "ex_g": g_norm}, mode=mode, need_weights=need_weights)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        denormalized_outputs = {key: tensor * g_std + g_mean for key, tensor in outputs.items()}
        if need_weights:
            return denormalized_outputs, None
        return denormalized_outputs

    def extract_hidden(self, data: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        f_norm, g_norm, _, _ = self._build_prompt(data)
        return self.backbone.encode_hidden(f_norm, g_norm)

    @staticmethod
    def _squeeze_sample_dim(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 5 and tensor.shape[1] == 1:
            return tensor[:, 0]
        return tensor

    @staticmethod
    def _cosine_bridge(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = 0.5 * math.pi * t
        theta_view = theta.view(theta.shape[0], 1, 1, 1)
        cos_theta = torch.cos(theta_view)
        sin_theta = torch.sin(theta_view)
        x_t = cos_theta * x0 + sin_theta * x1
        velocity_target = 0.5 * math.pi * (-sin_theta * x0 + cos_theta * x1)
        return x_t, velocity_target

    def compute_flow_matching_loss(self, target: torch.Tensor, cond: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        target = self._squeeze_sample_dim(target)
        cond = self._squeeze_sample_dim(cond)
        baseline = self._squeeze_sample_dim(baseline)

        if target.ndim != 4 or cond.ndim != 4 or baseline.ndim != 4:
            raise ValueError("target, cond, and baseline must be 4D tensors shaped [B, C, H, W]")

        batch_size = target.shape[0]
        dtype = target.dtype
        device = target.device

        residual_target = target - baseline
        t = torch.rand(batch_size, device=device, dtype=dtype)
        x_t, velocity_target = self._cosine_bridge(torch.randn_like(residual_target), residual_target, t)
        velocity_pred = self.flow_expert(x_t=x_t, cond=cond, t=t)

        time_weight = 1.0 + self.loss_time_weight * t.pow(self.loss_time_power)
        loss = (velocity_pred - velocity_target).pow(2).flatten(1).mean(dim=1)
        return time_weight * loss

    def sample_from_flow(
        self,
        cond: torch.Tensor,
        baseline: torch.Tensor,
        num_steps: int,
        num_samples: int,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, but got {num_steps}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, but got {num_samples}")

        cond = self._squeeze_sample_dim(cond)
        baseline = self._squeeze_sample_dim(baseline)

        batch_size = cond.shape[0]
        dtype = cond.dtype
        device = cond.device
        dt = 1.0 / num_steps
        sample_list = []

        for _ in range(num_samples):
            x = torch.randn(batch_size, baseline.shape[1], target_size[0], target_size[1], device=device, dtype=dtype)
            for step in range(num_steps):
                t_value = step / num_steps
                t = torch.full((batch_size,), t_value, device=device, dtype=dtype)
                v1 = self.flow_expert(x_t=x, cond=cond, t=t)
                x_mid = x + dt * v1

                t_next = torch.full((batch_size,), min(t_value + dt, 1.0), device=device, dtype=dtype)
                v2 = self.flow_expert(x_t=x_mid, cond=cond, t=t_next)
                x = x + 0.5 * dt * (v1 + v2)

            sample_list.append(x + baseline)

        return torch.stack(sample_list, dim=0).mean(dim=0)