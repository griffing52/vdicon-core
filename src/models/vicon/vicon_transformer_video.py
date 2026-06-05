from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vicon_latent_dicon import PretrainedViconLatentDICON, _extract_state_dict, _strip_prefix


class ProgressiveImageDecoder(nn.Module):
    """Decode a low-resolution transformer hidden map into a dense image."""

    def __init__(self, in_channels: int, decoder_channels: list[int], output_channels: int) -> None:
        super().__init__()
        if not decoder_channels:
            raise ValueError("decoder_channels must contain at least one channel size")

        layers: list[nn.Module] = []
        current_channels = in_channels
        for hidden_channels in decoder_channels:
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=max(1, min(8, hidden_channels)), num_channels=hidden_channels),
                    nn.SiLU(),
                ]
            )
            current_channels = hidden_channels
        layers.append(nn.Conv2d(current_channels, output_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        decoded = self.net(hidden)
        if decoded.shape[-2:] != target_size:
            decoded = F.interpolate(decoded, size=target_size, mode="bilinear", align_corners=False)
        return decoded


class ViconTransformerVideoModel(nn.Module):
    """LTX-style VICON adapter that uses the pretrained VICON transformer as the video core.

    The pretrained VICON checkpoint was trained with 7-channel 128x128 prompts. This
    wrapper lifts PDEArena's 2-channel velocity fields into that layout, runs the VICON
    transformer over the formatted prompt, then decodes the query hidden map into the
    requested 2-channel target resolution.
    """

    velocity_channel_indices = PretrainedViconLatentDICON.velocity_channel_indices

    def __init__(
        self,
        backbone: nn.Module,
        backbone_ckpt_path: str | Path | None,
        backbone_strict: bool,
        freeze_transformer: bool,
        freeze_backbone_adapters: bool,
        decoder_channels: list[int],
        output_channels: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_channels = output_channels

        if backbone_ckpt_path is not None:
            self._load_backbone_checkpoint(Path(backbone_ckpt_path), strict=backbone_strict)

        # The wrapper uses its own 512x512 decoder, so the original VICON patch
        # output projection is intentionally left out of the trainable path.
        self.backbone.post_proj.requires_grad_(False)

        if freeze_transformer:
            self.backbone.transformer.requires_grad_(False)
        if freeze_backbone_adapters:
            for name, parameter in self.backbone.named_parameters():
                if not name.startswith("transformer."):
                    parameter.requires_grad = False

        self.decoder = ProgressiveImageDecoder(
            in_channels=self.backbone.dim_token,
            decoder_channels=decoder_channels,
            output_channels=output_channels,
        )

    def _load_backbone_checkpoint(self, checkpoint_path: Path, strict: bool) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
                self.backbone.load_state_dict(candidate, strict=strict)
                load_error = None
                break
            except RuntimeError as err:
                load_error = err

        if load_error is not None:
            raise RuntimeError(f"Failed to load VICON backbone checkpoint from {checkpoint_path}") from load_error

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

    def _build_prompt(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ex_f = PretrainedViconLatentDICON.lift_to_pretrained_channels(self._resize_for_backbone(data["ex_f"]))
        ex_g = PretrainedViconLatentDICON.lift_to_pretrained_channels(self._resize_for_backbone(data["ex_g"]))
        qn_f = PretrainedViconLatentDICON.lift_to_pretrained_channels(self._resize_for_backbone(data["qn_f"]))
        dummy_label = torch.zeros_like(qn_f)

        f = torch.cat((ex_f, qn_f), dim=1)
        g = torch.cat((ex_g, dummy_label), dim=1)
        f_norm, _, _ = PretrainedViconLatentDICON._prompt_normalization(f)
        g_norm, g_mean, g_std = PretrainedViconLatentDICON._prompt_normalization(ex_g)
        g_norm = torch.cat((g_norm, dummy_label), dim=1)
        return f_norm, g_norm, g_mean, g_std

    def _run_transformer_hidden(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_norm, g_norm, g_mean, g_std = self._build_prompt(data)
        hidden = self.backbone.encode_hidden(f_norm, g_norm)
        return hidden[:, 0], g_mean, g_std

    def _velocity_stats_at_target(
        self, g_mean: torch.Tensor, g_std: torch.Tensor, target_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = g_mean[:, :1, self.velocity_channel_indices, :, :].flatten(0, 1)
        std = g_std[:, :1, self.velocity_channel_indices, :, :].flatten(0, 1)
        if mean.shape[-2:] != target_size:
            mean = F.interpolate(mean, size=target_size, mode="bilinear", align_corners=False)
            std = F.interpolate(std, size=target_size, mode="bilinear", align_corners=False)
        return mean.unsqueeze(1), std.unsqueeze(1)

    def forward(self, data: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
        target_size = data["qn_f"].shape[-2:]
        hidden, g_mean, g_std = self._run_transformer_hidden(data)
        qn_pred_norm = self.decoder(hidden, target_size=target_size).unsqueeze(1)
        mean, std = self._velocity_stats_at_target(g_mean, g_std, target_size=target_size)
        return {"qn_pred": qn_pred_norm * std + mean}

    def predict_backbone(
        self, data: dict[str, torch.Tensor], mode: str, need_weights: bool
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], None]:
        outputs = self.forward(data=data, mode=mode)
        if need_weights:
            return outputs, None
        return outputs
