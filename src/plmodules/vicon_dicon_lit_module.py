from __future__ import annotations

import einops
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from optree import PyTree
from torchmetrics import MeanMetric, MetricCollection

import src.utils.icon_core_utils as cu
from src.plmodules.base_lit_module import BaseLitModule


class ViconDiconLitModule(BaseLitModule):
    def __init__(self, cfg: DictConfig, loss_icon_weight: float, loss_fm_weight: float, eval_flow_steps: int, eval_num_samples: int) -> None:
        super().__init__(cfg)

        self.loss_icon_weight = loss_icon_weight
        self.loss_fm_weight = loss_fm_weight
        self.eval_flow_steps = eval_flow_steps
        self.eval_num_samples = eval_num_samples

        self.train_metrics = MeanMetric()

        self.metric_names = [
            "loss",
            "loss_icon",
            "loss_fm",
            "quest_qoi_v_icon",
            "quest_qoi_v_dicon",
            "quest_qoi_v_gain",
        ]

        self.valid_metrics = torch.nn.ModuleDict(
            {
                self.cfg.data.valid[key].name: MetricCollection({k: MeanMetric() for k in self.metric_names})
                for key in self.cfg.data.valid
            }
        )

    def get_trainable_networks(self):
        return self.net

    def _prompt_normalization(self, x: torch.Tensor):
        mean = x.mean(dim=(1, 3, 4), keepdim=True)
        std = x.std(dim=(1, 3, 4), keepdim=True) + 1e-5
        x_normalized = (x - mean) / std
        return x_normalized, mean, std

    def network_inference(self, data: PyTree):
        dummy_label = torch.zeros_like(data["ex_g"][:, -1:, :, :, :])
        g = data["ex_g"]
        f = torch.cat((data["ex_f"], data["qn_f"]), dim=1)
        f_norm, _, _ = self._prompt_normalization(f)
        g_norm, g_mean, g_std = self._prompt_normalization(g)
        g_norm = torch.cat((g_norm, dummy_label), dim=1)

        outputs = self._model_forward(f_norm, g_norm)
        denormalized_outputs = {}
        for key, tensor in outputs.items():
            denormalized_outputs[key] = tensor * g_std + g_mean

        return denormalized_outputs

    def _get_ground_truth_all(self, batch: PyTree):
        return torch.cat((batch["data"]["ex_g"], batch["label"]), dim=1)

    def _get_pred_all(self, outputs: dict):
        return torch.cat([outputs["ex_pred"], outputs["qn_pred"]], dim=1)

    def _get_pred_qn(self, outputs: dict):
        return outputs["qn_pred"]

    def _loss_function(self, pred: torch.Tensor, target: torch.Tensor):
        return F.mse_loss(pred, target)

    def _loss_all(self, batch: PyTree) -> dict[str, torch.Tensor]:
        outputs = self.network_inference(batch["data"])
        all_pred = self._get_pred_all(outputs)
        all_ground_truth = self._get_ground_truth_all(batch)
        loss_icon = (all_pred - all_ground_truth).pow(2).flatten(1).mean(dim=1)

        qn_pred = self._get_pred_qn(outputs)[:, 0]
        qn_target = batch["label"][:, 0]
        loss_fm = self.net.compute_flow_matching_loss(target=qn_target, cond=qn_pred)

        loss = self.loss_icon_weight * loss_icon + self.loss_fm_weight * loss_fm
        return {
            "loss": loss,
            "loss_icon": loss_icon,
            "loss_fm": loss_fm,
        }

    def _error_all(self, batch: PyTree) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.network_inference(batch["data"])
        all_pred = self._get_pred_all(outputs)
        all_ground_truth = self._get_ground_truth_all(batch)
        error = all_pred - all_ground_truth
        return all_pred, error

    def _sample_qn(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        qn_pred = self._get_pred_qn(outputs)[:, 0]
        qn_sample = self.net.sample_from_flow(cond=qn_pred, num_steps=self.eval_flow_steps, num_samples=self.eval_num_samples)
        return qn_sample[:, None]

    def get_preds(self, data: PyTree) -> dict[str, torch.Tensor]:
        outputs = self.network_inference(data)
        return {
            "quest_qoi_v_icon": self._get_pred_qn(outputs),
            "quest_qoi_v_dicon": self._sample_qn(outputs),
        }

    def get_errors(self, preds: dict[str, torch.Tensor], batch: PyTree) -> dict[str, torch.Tensor]:
        return {
            "quest_qoi_v_icon": torch.abs(preds["quest_qoi_v_icon"] - batch["label"]).flatten(1).mean(dim=1),
            "quest_qoi_v_dicon": torch.abs(preds["quest_qoi_v_dicon"] - batch["label"]).flatten(1).mean(dim=1),
            "quest_qoi_v_gain": torch.abs(preds["quest_qoi_v_icon"] - batch["label"]).flatten(1).mean(dim=1)
            - torch.abs(preds["quest_qoi_v_dicon"] - batch["label"]).flatten(1).mean(dim=1),
        }

    def _error_qn(self, batch: PyTree) -> torch.Tensor:
        outputs = self.network_inference(batch["data"])
        qn_pred = self._get_pred_qn(outputs)[:, 0]
        qn_sample = self.net.sample_from_flow(cond=qn_pred, num_steps=self.eval_flow_steps, num_samples=self.eval_num_samples)
        return qn_sample - batch["label"][:, 0]

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics.values():
            metrics.reset()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        losses = self._loss_all(batch)

        self.train_metrics(losses["loss"])
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=True)
        self.log("train/loss_icon", losses["loss_icon"].mean(), on_step=True, on_epoch=True)
        self.log("train/loss_fm", losses["loss_fm"].mean(), on_step=True, on_epoch=True)

        return losses["loss"].mean()

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        losses = self._loss_all(batch)
        outputs = self.network_inference(batch["data"])

        preds = {
            "quest_qoi_v_icon": self._get_pred_qn(outputs),
            "quest_qoi_v_dicon": self._sample_qn(outputs),
        }
        errors = {
            "quest_qoi_v_icon": torch.abs(preds["quest_qoi_v_icon"] - batch["label"]).flatten(1).mean(dim=1),
            "quest_qoi_v_dicon": torch.abs(preds["quest_qoi_v_dicon"] - batch["label"]).flatten(1).mean(dim=1),
        }
        errors["quest_qoi_v_gain"] = errors["quest_qoi_v_icon"] - errors["quest_qoi_v_dicon"]

        metrics = {
            "loss": losses["loss"].mean(),
            "loss_icon": losses["loss_icon"].mean(),
            "loss_fm": losses["loss_fm"].mean(),
            **errors,
        }

        valid_name = cu.get_dataset_name(self.cfg.data.valid, dataloader_idx)
        for metric_name in self.metric_names:
            self.valid_metrics[valid_name][metric_name].update(metrics[metric_name])

        for metric_name in self.metric_names:
            self.log(
                f"{valid_name}/{metric_name}",
                self.valid_metrics[valid_name][metric_name],
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )

        return {"preds": preds, "errors": errors, "metrics": metrics}