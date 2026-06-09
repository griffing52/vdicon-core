from __future__ import annotations

from collections.abc import Mapping

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .ltx_vicon import _resolve_torch_dtype, _to_plain_dict


class SpatialTokenDecoder(nn.Module):
    """Decode transformer tokens on a patch grid into a dense velocity residual."""

    def __init__(self, token_channels: int, hidden_channels: int, output_channels: int, num_blocks: int) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")

        layers: list[nn.Module] = [
            nn.Conv2d(token_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, hidden_channels)), num_channels=hidden_channels),
            nn.SiLU(),
        ]
        for _ in range(num_blocks - 1):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=max(1, min(8, hidden_channels)), num_channels=hidden_channels),
                    nn.SiLU(),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor, height_tokens: int, width_tokens: int, target_size: tuple[int, int]) -> torch.Tensor:
        feature = einops.rearrange(
            tokens,
            "batch (height width) channel -> batch channel height width",
            height=height_tokens,
            width=width_tokens,
        )
        decoded = self.net(feature)
        if decoded.shape[-2:] != target_size:
            decoded = F.interpolate(decoded, size=target_size, mode="bilinear", align_corners=False)
        return decoded.unsqueeze(1)


class LTXViconVideoModelV2(nn.Module):
    """Less collapse-prone LTX-VICON adapter for direct next-frame prediction.

    Compared with the first LTX wrapper, this version uses the current query field as
    the spatial hidden-state stream and decodes a residual on top of that query field.
    A zero decoder therefore copies qn_f instead of returning the prompt mean.
    """

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
        context_tokens: int,
        timestep: int,
        train_timestep_min: int | None,
        train_timestep_max: int | None,
        random_train_timestep: bool,
        freeze_transformer: bool,
        train_last_n_layers: int,
        gradient_checkpointing: bool,
        decoder_hidden_channels: int,
        decoder_num_blocks: int,
        residual_scale: float,
        output_revision: str,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if context_tokens <= 0:
            raise ValueError(f"context_tokens must be positive, got {context_tokens}")
        if train_timestep_min is not None and train_timestep_max is not None and train_timestep_min > train_timestep_max:
            raise ValueError("train_timestep_min must be <= train_timestep_max")
        if output_revision != "target_stats_residual":
            raise ValueError(
                "LTXViconVideoModelV2 now requires output_revision='target_stats_residual'. "
                "Older checkpoints were trained with a mismatched residual normalization and should be retrained."
            )

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.subfolder = subfolder
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.token_channels = token_channels
        self.patch_size = patch_size
        self.context_tokens = context_tokens
        self.timestep = timestep
        self.train_timestep_min = train_timestep_min
        self.train_timestep_max = train_timestep_max
        self.random_train_timestep = random_train_timestep
        self.residual_scale = residual_scale
        self.output_revision = output_revision

        patch_dim = input_channels * patch_size * patch_size
        self.input_patch_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, token_channels),
            nn.SiLU(),
            nn.Linear(token_channels, token_channels),
        )
        self.query_norm = nn.LayerNorm(token_channels)
        self.condition_norm = nn.LayerNorm(token_channels)

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
        self.decoder = SpatialTokenDecoder(
            token_channels=token_channels,
            hidden_channels=decoder_hidden_channels,
            output_channels=output_channels,
            num_blocks=decoder_num_blocks,
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
                "diffusers is required for LTXViconVideoModelV2. Run `uv sync --extra cu126` "
                "or install the project dependencies."
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

    @staticmethod
    def _build_context_video(f_norm: torch.Tensor, g_norm: torch.Tensor) -> torch.Tensor:
        ex_f_norm = f_norm[:, :-1]
        qn_f_norm = f_norm[:, -1:]
        examples = torch.stack((ex_f_norm, g_norm), dim=2)
        examples = einops.rearrange(examples, "batch examples pair channel height width -> batch (examples pair) channel height width")
        return torch.cat((examples, qn_f_norm), dim=1)

    def _normalize_data(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f = torch.cat((data["ex_f"], data["qn_f"]), dim=1)
        g = data["ex_g"]
        f_norm, _, _ = self._prompt_normalization(f)
        g_norm, g_mean, g_std = self._prompt_normalization(g)
        qn_f_query_norm = f_norm[:, -1:]
        qn_f_output_norm = (data["qn_f"] - g_mean[:, :, : self.output_channels]) / g_std[:, :, : self.output_channels]
        context = self._build_context_video(f_norm=f_norm, g_norm=g_norm)
        return context, g_mean, g_std, qn_f_query_norm, qn_f_output_norm, data["qn_f"]

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

    def _resize_condition_sequence(self, context_tokens: torch.Tensor) -> torch.Tensor:
        token_count = context_tokens.shape[1]
        if token_count == self.context_tokens:
            return context_tokens

        # Avoid CUDA adaptive_avg_pool1d backward on long token sequences; that
        # kernel can exceed shared-memory launch bounds for 512x512 prompts.
        indices = torch.linspace(
            0,
            token_count - 1,
            steps=self.context_tokens,
            device=context_tokens.device,
        ).round().long()
        return context_tokens.index_select(dim=1, index=indices)

    def _build_condition_tokens(self, context: torch.Tensor) -> torch.Tensor:
        context_tokens, _, _ = self._patchify_video(context)
        context_tokens = self.condition_norm(context_tokens)
        context_tokens = self._resize_condition_sequence(context_tokens)
        return self.condition_proj(context_tokens)

    def _build_timestep(self, batch_size: int, mode: str, device: torch.device) -> torch.Tensor:
        if (
            mode == "train"
            and self.random_train_timestep
            and self.train_timestep_min is not None
            and self.train_timestep_max is not None
        ):
            return torch.randint(
                low=self.train_timestep_min,
                high=self.train_timestep_max + 1,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
        return torch.full((batch_size,), self.timestep, device=device, dtype=torch.long)

    def _transformer_dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    @staticmethod
    def _spatial_variation(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x[:, 0]
        return x.flatten(2).std(dim=-1).mean()

    def _forward_impl(self, data: dict[str, torch.Tensor], mode: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        context, g_mean, g_std, qn_f_query_norm, qn_f_output_norm, qn_f = self._normalize_data(data)
        batch_size = context.shape[0]
        query_tokens, height_tokens, width_tokens = self._patchify_video(qn_f_query_norm)
        hidden_states = self.query_norm(query_tokens)
        encoder_hidden_states = self._build_condition_tokens(context)

        transformer_dtype = self._transformer_dtype()
        timestep = self._build_timestep(batch_size=batch_size, mode=mode, device=context.device)
        encoder_attention_mask = torch.ones(
            batch_size,
            encoder_hidden_states.shape[1],
            device=context.device,
            dtype=encoder_hidden_states.dtype,
        )
        transformer_output = self.transformer(
            hidden_states=hidden_states.to(transformer_dtype),
            encoder_hidden_states=encoder_hidden_states.to(transformer_dtype),
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask.to(transformer_dtype),
            num_frames=1,
            height=height_tokens,
            width=width_tokens,
            return_dict=True,
        ).sample
        transformer_output = transformer_output.to(self.decoder.net[0].weight.dtype)
        delta_norm = self.decoder(
            transformer_output,
            height_tokens=height_tokens,
            width_tokens=width_tokens,
            target_size=qn_f_output_norm.shape[-2:],
        )
        qn_pred_norm = qn_f_output_norm + self.residual_scale * delta_norm
        qn_pred = qn_pred_norm * g_std[:, :, : self.output_channels] + g_mean[:, :, : self.output_channels]

        diagnostics = {
            "query_token_std": query_tokens.detach().float().std(),
            "hidden_token_std": transformer_output.detach().float().std(),
            "delta_spatial_std": self._spatial_variation(delta_norm.detach().float()),
            "pred_spatial_std": self._spatial_variation(qn_pred.detach().float()),
            "input_spatial_std": self._spatial_variation(qn_f.detach().float()),
            "output_base_spatial_std": self._spatial_variation(
                (qn_f_output_norm * g_std[:, :, : self.output_channels] + g_mean[:, :, : self.output_channels]).detach().float()
            ),
            "timestep_mean": timestep.detach().float().mean(),
        }
        return {"qn_pred": qn_pred}, diagnostics

    def forward(self, data: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
        outputs, _ = self._forward_impl(data=data, mode=mode)
        return outputs

    @torch.no_grad()
    def diagnose(self, data: dict[str, torch.Tensor], mode: str = "test") -> dict[str, float]:
        was_training = self.training
        self.eval()
        outputs, diagnostics = self._forward_impl(data=data, mode=mode)
        if was_training:
            self.train()
        diagnostics["pred_min"] = outputs["qn_pred"].detach().float().min()
        diagnostics["pred_max"] = outputs["qn_pred"].detach().float().max()
        diagnostics["pred_mean"] = outputs["qn_pred"].detach().float().mean()
        return {key: float(value.detach().cpu()) for key, value in diagnostics.items()}

    def predict_backbone(
        self, data: dict[str, torch.Tensor], mode: str, need_weights: bool
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], None]:
        outputs = self.forward(data=data, mode=mode)
        if need_weights:
            return outputs, None
        return outputs
