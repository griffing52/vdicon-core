from __future__ import annotations

from collections.abc import Mapping

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


def _resolve_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None or dtype_name == "none":
        return None
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype '{dtype_name}'")
    return dtype_map[dtype_name]


def _to_plain_dict(value: Mapping | DictConfig) -> dict:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


class LTXViconVideoModel(nn.Module):
    """Adapt LTXVideoTransformer3DModel to VICON-style next-frame PDE prediction."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        subfolder: str,
        use_pretrained: bool,
        torch_dtype: str | None,
        transformer_kwargs: Mapping | DictConfig,
        input_channels: int,
        output_channels: int,
        token_channels: int,
        patch_size: int,
        output_frames: int,
        context_tokens: int,
        timestep: int,
        freeze_transformer: bool,
        train_last_n_layers: int,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        if output_frames != 1:
            raise ValueError("Only output_frames=1 is currently supported for PDEArena next-frame prediction")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if context_tokens <= 0:
            raise ValueError(f"context_tokens must be positive, got {context_tokens}")

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.subfolder = subfolder
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.token_channels = token_channels
        self.patch_size = patch_size
        self.output_frames = output_frames
        self.context_tokens = context_tokens
        self.timestep = timestep

        patch_dim = input_channels * patch_size * patch_size
        output_patch_dim = output_channels * patch_size * patch_size
        self.input_patch_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, token_channels),
            nn.SiLU(),
            nn.Linear(token_channels, token_channels),
        )
        self.query_tokens = nn.Parameter(torch.randn(1, output_frames, token_channels) / token_channels**0.5)
        self.condition_norm = nn.LayerNorm(token_channels)
        self.output_patch_proj = nn.Sequential(
            nn.LayerNorm(token_channels),
            nn.Linear(token_channels, token_channels),
            nn.SiLU(),
            nn.Linear(token_channels, output_patch_dim),
        )

        self.transformer = self._build_transformer(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            use_pretrained=use_pretrained,
            torch_dtype=torch_dtype,
            transformer_kwargs=transformer_kwargs,
        )
        self.caption_channels = self.transformer.config.caption_channels
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(token_channels),
            nn.Linear(token_channels, self.caption_channels),
        )

        if gradient_checkpointing and hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        elif gradient_checkpointing and hasattr(self.transformer, "gradient_checkpointing"):
            self.transformer.gradient_checkpointing = True

        self._configure_trainable_transformer(freeze_transformer=freeze_transformer, train_last_n_layers=train_last_n_layers)

    def _build_transformer(
        self,
        pretrained_model_name_or_path: str,
        subfolder: str,
        use_pretrained: bool,
        torch_dtype: str | None,
        transformer_kwargs: Mapping | DictConfig,
    ) -> nn.Module:
        try:
            from diffusers import LTXVideoTransformer3DModel
        except ImportError as err:
            raise ImportError(
                "diffusers is required for LTXViconVideoModel. Install it with `uv sync --extra cu126` "
                "after the pyproject dependency update, or `uv add diffusers`."
            ) from err

        dtype = _resolve_torch_dtype(torch_dtype)
        kwargs = _to_plain_dict(transformer_kwargs)
        if use_pretrained:
            return LTXVideoTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder=subfolder,
                torch_dtype=dtype,
            )
        return LTXVideoTransformer3DModel(**kwargs)

    def _configure_trainable_transformer(self, freeze_transformer: bool, train_last_n_layers: int) -> None:
        if train_last_n_layers < 0:
            raise ValueError(f"train_last_n_layers must be >= 0, got {train_last_n_layers}")
        if not freeze_transformer:
            self.transformer.requires_grad_(True)
            return

        self.transformer.requires_grad_(False)
        if train_last_n_layers == 0:
            return
        blocks = self.transformer.transformer_blocks
        for block in blocks[-train_last_n_layers:]:
            block.requires_grad_(True)
        self.transformer.norm_out.requires_grad_(True)
        self.transformer.proj_out.requires_grad_(True)

    @staticmethod
    def _prompt_normalization(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(1, 3, 4), keepdim=True)
        std = x.std(dim=(1, 3, 4), keepdim=True) + 1e-5
        return (x - mean) / std, mean, std

    def _normalize_data(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f = torch.cat((data["ex_f"], data["qn_f"]), dim=1)
        g = data["ex_g"]
        f_norm, _, _ = self._prompt_normalization(f)
        g_norm, g_mean, g_std = self._prompt_normalization(g)
        context = self._build_context_video(f_norm=f_norm, g_norm=g_norm)
        return context, g_mean, g_std, f_norm[:, -1:]

    @staticmethod
    def _build_context_video(f_norm: torch.Tensor, g_norm: torch.Tensor) -> torch.Tensor:
        ex_f_norm = f_norm[:, :-1]
        qn_f_norm = f_norm[:, -1:]
        examples = torch.stack((ex_f_norm, g_norm), dim=2)
        examples = einops.rearrange(examples, "batch examples pair channel height width -> batch (examples pair) channel height width")
        return torch.cat((examples, qn_f_norm), dim=1)

    def _check_image_size(self, height: int, width: int) -> None:
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"Image size {(height, width)} must be divisible by patch_size={self.patch_size}")

    def _patchify_video(self, video: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, _, height, width = video.shape
        self._check_image_size(height, width)
        height_tokens = height // self.patch_size
        width_tokens = width // self.patch_size
        patches = einops.rearrange(
            video,
            "batch frames channel (height patch_h) (width patch_w) -> batch (frames height width) (channel patch_h patch_w)",
            patch_h=self.patch_size,
            patch_w=self.patch_size,
        )
        return self.input_patch_proj(patches), height_tokens, width_tokens

    def _build_condition_tokens(self, context: torch.Tensor) -> torch.Tensor:
        context_tokens, _, _ = self._patchify_video(context)
        context_tokens = self.condition_norm(context_tokens)
        pooled = F.adaptive_avg_pool1d(
            einops.rearrange(context_tokens, "batch tokens channel -> batch channel tokens"),
            self.context_tokens,
        )
        pooled = einops.rearrange(pooled, "batch channel tokens -> batch tokens channel")
        return self.condition_proj(pooled)

    def _build_query_tokens(self, batch_size: int, height_tokens: int, width_tokens: int, device: torch.device) -> torch.Tensor:
        frame_tokens = height_tokens * width_tokens
        query = self.query_tokens.to(device=device).expand(batch_size, -1, -1)
        query = einops.repeat(query, "batch frames channel -> batch (frames tokens) channel", tokens=frame_tokens)
        return query

    def _unpatchify_image(self, tokens: torch.Tensor, height_tokens: int, width_tokens: int) -> torch.Tensor:
        patches = self.output_patch_proj(tokens)
        image = einops.rearrange(
            patches,
            "batch (height width) (channel patch_h patch_w) -> batch 1 channel (height patch_h) (width patch_w)",
            height=height_tokens,
            width=width_tokens,
            channel=self.output_channels,
            patch_h=self.patch_size,
            patch_w=self.patch_size,
        )
        return image

    def _transformer_dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    def forward(self, data: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
        """Predict the next query frame from VICON-style context data."""
        context, g_mean, g_std, qn_f_norm = self._normalize_data(data)
        batch_size = context.shape[0]
        _, height_tokens, width_tokens = self._patchify_video(qn_f_norm)
        hidden_states = self._build_query_tokens(batch_size, height_tokens, width_tokens, context.device)
        encoder_hidden_states = self._build_condition_tokens(context)
        transformer_dtype = self._transformer_dtype()
        timestep = torch.full((batch_size,), self.timestep, device=context.device, dtype=torch.long)
        encoder_attention_mask = torch.ones(
            batch_size,
            encoder_hidden_states.shape[1],
            device=context.device,
            dtype=encoder_hidden_states.dtype,
        )
        output = self.transformer(
            hidden_states=hidden_states.to(transformer_dtype),
            encoder_hidden_states=encoder_hidden_states.to(transformer_dtype),
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask.to(transformer_dtype),
            num_frames=self.output_frames,
            height=height_tokens,
            width=width_tokens,
            return_dict=True,
        ).sample
        qn_pred_norm = self._unpatchify_image(output.to(self.output_patch_proj[1].weight.dtype), height_tokens, width_tokens)
        return {"qn_pred": qn_pred_norm * g_std + g_mean}

    def predict_backbone(
        self, data: dict[str, torch.Tensor], mode: str, need_weights: bool
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], None]:
        """Compatibility wrapper matching the VICON model API."""
        outputs = self.forward(data=data, mode=mode)
        if need_weights:
            return outputs, None
        return outputs
