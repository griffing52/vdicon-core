from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from optree import PyTree
from torchmetrics import MeanMetric, MetricCollection

import src.utils.icon_core_utils as cu
from src.plmodules.base_lit_module import BaseLitModule


class LTXViconLitModule(BaseLitModule):
    """Lightning module for supervised LTX-VICON next-frame prediction."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.train_metrics = MeanMetric()
        self.metric_names = ["loss", "quest_qoi_v"]
        self.valid_metrics = torch.nn.ModuleDict(
            {
                self.cfg.data.valid[key].name: MetricCollection({metric_name: MeanMetric() for metric_name in self.metric_names})
                for key in self.cfg.data.valid
            }
        )

    def get_trainable_networks(self) -> torch.nn.Module:
        """Return the wrapped LTX-VICON network for optimization."""
        return self.net

    def _loss_all(self, batch: PyTree) -> dict[str, torch.Tensor]:
        preds = self.net(data=batch["data"], mode="train")
        loss = F.mse_loss(preds["qn_pred"], batch["label"], reduction="none").flatten(1).mean(dim=1)
        return {"loss": loss}

    def get_preds(self, data: PyTree) -> dict[str, torch.Tensor]:
        """Return predictions for validation and visualization."""
        preds = self.net(data=data, mode="test")
        return {"quest_qoi_v": preds["qn_pred"]}

    def get_errors(self, preds: dict[str, torch.Tensor], batch: PyTree) -> dict[str, torch.Tensor]:
        """Compute absolute qoi error for validation metrics."""
        return {"quest_qoi_v": torch.abs(preds["quest_qoi_v"] - batch["label"]).flatten(1).mean(dim=1)}

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics.values():
            metrics.reset()

    def training_step(self, batch: PyTree, batch_idx: int) -> torch.Tensor:
        losses = self._loss_all(batch)
        self.train_metrics(losses["loss"])
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=True)
        return losses["loss"].mean()

    def validation_step(self, batch: PyTree, batch_idx: int, dataloader_idx: int = 0) -> dict:
        losses = self._loss_all(batch)
        preds = self.get_preds(batch["data"])
        errors = self.get_errors(preds, batch)
        metrics = {"loss": losses["loss"].mean(), **errors}

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
