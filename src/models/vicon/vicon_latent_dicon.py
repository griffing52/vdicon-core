from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "net", "backbone", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(key, str) and torch.is_tensor(value) for key, value in checkpoint.items()):
            return checkpoint

    raise ValueError("Expected a checkpoint dictionary or a raw state dict")


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


class LatentImageFlowMatchingExpert(nn.Module):
    """Conditional velocity field that consumes a VICON hidden map."""

    def __init__(self, target_channels: int, cond_channels: int, hidden_channels: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, but got {num_layers}")

        self.cond_proj = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
        )

        layers: list[nn.Module] = []
        in_channels = target_channels + hidden_channels + 1
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(num_groups=max(1, min(8, hidden_channels)), num_channels=hidden_channels))
            layers.append(nn.SiLU())
            in_channels = hidden_channels
        layers.append(nn.Conv2d(hidden_channels, target_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim == 5 and x_t.shape[1] == 1:
            x_t = x_t[:, 0]
        if cond.ndim == 5 and cond.shape[1] == 1:
            cond = cond[:, 0]
        if cond.shape[-2:] != x_t.shape[-2:]:
            cond = F.interpolate(cond, size=x_t.shape[-2:], mode="bilinear", align_corners=False)

        cond = self.cond_proj(cond)

        if t.ndim == 1:
            t = t[:, None, None, None]
        elif t.ndim == 2:
            t = t[:, :, None, None]
        t_feature = torch.broadcast_to(t, x_t.shape[:1] + (1,) + x_t.shape[2:])

        model_input = torch.cat([x_t, cond, t_feature], dim=1)
        return self.net(model_input)


class ViconLatentDICON(nn.Module):
    """VICON encoder plus latent flow-matching head."""

    def __init__(
        self,
        backbone: nn.Module,
        flow_expert: nn.Module,
        backbone_ckpt_path: str | Path | None = None,
        freeze_backbone: bool = False,
        backbone_strict: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.flow_expert = flow_expert
        self.freeze_backbone = freeze_backbone

        if backbone_ckpt_path is not None:
            checkpoint = torch.load(Path(backbone_ckpt_path), map_location="cpu")
            state_dict = _extract_state_dict(checkpoint)
            load_attempts = [
                state_dict,
                _strip_prefix(state_dict, "net."),
                _strip_prefix(state_dict, "backbone."),
                _strip_prefix(state_dict, "model."),
                _strip_prefix(state_dict, "module."),
            ]

            load_error: RuntimeError | None = None
            for candidate in load_attempts:
                try:
                    self.backbone.load_state_dict(candidate, strict=backbone_strict)
                    load_error = None
                    break
                except RuntimeError as err:
                    load_error = err

            if load_error is not None:
                raise RuntimeError(f"Failed to load backbone checkpoint from {backbone_ckpt_path}") from load_error

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)

    @staticmethod
    def _prompt_normalization(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(1, 3, 4), keepdim=True)
        std = x.std(dim=(1, 3, 4), keepdim=True) + 1e-5
        x_normalized = (x - mean) / std
        return x_normalized, mean, std

    def _build_prompt(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def predict_backbone(
        self, data: dict[str, torch.Tensor], mode: str, need_weights: bool
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], None]:
        """Run the pretrained VICON backbone and return velocity-channel predictions."""
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

    def compute_flow_matching_loss(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if target.ndim == 5 and target.shape[1] == 1:
            target = target[:, 0]
        if cond.ndim == 5 and cond.shape[1] == 1:
            cond = cond[:, 0]

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

    def sample_from_flow(
        self,
        cond: torch.Tensor,
        num_steps: int,
        num_samples: int,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, but got {num_steps}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, but got {num_samples}")

        if cond.ndim == 5 and cond.shape[1] == 1:
            cond = cond[:, 0]

        batch_size = cond.shape[0]
        dtype = cond.dtype
        device = cond.device
        dt = 1.0 / num_steps
        sample_list = []
        target_channels = self.flow_expert.net[-1].out_channels

        for _ in range(num_samples):
            x = torch.randn(batch_size, target_channels, target_size[0], target_size[1], device=device, dtype=dtype)
            for step in range(num_steps):
                t_value = (step + 0.5) / num_steps
                t = torch.full((batch_size,), t_value, device=device, dtype=dtype)
                t_view = t.view(batch_size, 1, 1, 1)
                velocity = self.flow_expert(x_t=x, cond=cond, t=t_view)
                x = x + dt * velocity
            sample_list.append(x)

        return torch.stack(sample_list, dim=0).mean(dim=0)


class PretrainedViconLatentDICON(ViconLatentDICON):
    """Latent DICON wrapper for the released 7-channel pretrained VICON."""

    velocity_channel_indices = (1, 2)

    def _backbone_image_size(self) -> tuple[int, int]:
        size = self.backbone.patch_resolution * self.backbone.patch_num_in
        return size, size

    def _resize_for_backbone(self, tensor: torch.Tensor) -> torch.Tensor:
        target_size = self._backbone_image_size()
        if tensor.shape[-2:] == target_size:
            return tensor
        batch_size, pair_count = tensor.shape[:2]
        tensor = tensor.flatten(0, 1)
        tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
        return tensor.unflatten(0, (batch_size, pair_count))

    @staticmethod
    def _resize_outputs(outputs: dict[str, torch.Tensor], target_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        resized_outputs = {}
        for key, tensor in outputs.items():
            if tensor.shape[-2:] == target_size:
                resized_outputs[key] = tensor
                continue
            batch_size, pair_count = tensor.shape[:2]
            tensor = tensor.flatten(0, 1)
            tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
            resized_outputs[key] = tensor.unflatten(0, (batch_size, pair_count))
        return resized_outputs

    @staticmethod
    def _build_node_type(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        node_type = torch.ones((1, 1, 1, height, width), device=device, dtype=dtype)
        node_type[..., 0, :] = 0
        node_type[..., -1, :] = 0
        node_type[..., :, 0] = 0
        node_type[..., :, -1] = 0
        return node_type

    @staticmethod
    def _finite_difference_vorticity(velocity: torch.Tensor) -> torch.Tensor:
        if velocity.ndim != 5 or velocity.shape[2] != 2:
            raise ValueError(f"Expected velocity shaped [B, P, 2, H, W], got {tuple(velocity.shape)}")

        u = velocity[:, :, 0]
        v = velocity[:, :, 1]
        du_dy = torch.gradient(u, dim=(-2, -1))[0]
        dv_dx = torch.gradient(v, dim=(-2, -1))[1]
        return (dv_dx - du_dy).unsqueeze(2)

    @classmethod
    def lift_to_pretrained_channels(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Lift 2-channel velocity fields to the 7-channel pretrained VICON layout."""
        if tensor.ndim != 5:
            raise ValueError(f"Expected tensor shaped [B, P, C, H, W], got {tuple(tensor.shape)}")
        if tensor.shape[2] == 7:
            return tensor
        if tensor.shape[2] != 2:
            raise ValueError(f"Expected 2 or 7 channels, got {tensor.shape[2]}")

        density = torch.zeros_like(tensor[:, :, :1])
        pressure = torch.zeros_like(tensor[:, :, :1])
        scalar = torch.zeros_like(tensor[:, :, :1])
        vorticity = cls._finite_difference_vorticity(tensor)
        node_type = cls._build_node_type(tensor.shape[-2], tensor.shape[-1], tensor.device, tensor.dtype)
        node_type = node_type.expand(tensor.shape[0], tensor.shape[1], -1, -1, -1)

        return torch.cat(
            [
                density,
                tensor[:, :, :1],
                tensor[:, :, 1:2],
                pressure,
                vorticity,
                scalar,
                node_type,
            ],
            dim=2,
        )

    @staticmethod
    def _prompt_normalization(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(1, 3, 4), keepdim=True)
        std = x.std(dim=(1, 3, 4), keepdim=True) + 1e-5
        if x.shape[2] == 7:
            mean[:, :, -1] = 0
            std[:, :, -1] = 1
        x_normalized = (x - mean) / std
        return x_normalized, mean, std

    @classmethod
    def _select_velocity_channels(cls, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: tensor[:, :, cls.velocity_channel_indices, :, :] for key, tensor in outputs.items()}

    def _build_prompt(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.lift_to_pretrained_channels(self._resize_for_backbone(data["ex_g"]))
        qn_f = self.lift_to_pretrained_channels(self._resize_for_backbone(data["qn_f"]))
        ex_f = self.lift_to_pretrained_channels(self._resize_for_backbone(data["ex_f"]))
        dummy_label = torch.zeros_like(qn_f)
        f = torch.cat((ex_f, qn_f), dim=1)
        f_norm, _, _ = self._prompt_normalization(f)
        g_norm, g_mean, g_std = self._prompt_normalization(g)
        g_norm = torch.cat((g_norm, dummy_label), dim=1)
        return f_norm, g_norm, g_mean, g_std

    def predict_backbone(
        self, data: dict[str, torch.Tensor], mode: str, need_weights: bool
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], None]:
        """Run the pretrained VICON backbone and return velocity-channel predictions."""
        f_norm, g_norm, g_mean, g_std = self._build_prompt(data)
        outputs = self._run_backbone({"ex_f": f_norm, "ex_g": g_norm}, mode=mode, need_weights=need_weights)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        denormalized_outputs = {key: tensor * g_std + g_mean for key, tensor in outputs.items()}
        denormalized_outputs = self._select_velocity_channels(denormalized_outputs)
        denormalized_outputs = self._resize_outputs(denormalized_outputs, target_size=data["qn_f"].shape[-2:])
        if need_weights:
            return denormalized_outputs, None
        return denormalized_outputs
