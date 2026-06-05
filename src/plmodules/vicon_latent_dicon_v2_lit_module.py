from __future__ import annotations

import torch
from omegaconf import DictConfig
from optree import PyTree
from torchmetrics import MeanMetric, MetricCollection

import src.utils.icon_core_utils as cu
from src.plmodules.base_lit_module import BaseLitModule


class ViconLatentDiconV2LitModule(BaseLitModule):
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

    def _get_ground_truth_all(self, batch: PyTree):
        return torch.cat((batch["data"]["ex_g"], batch["label"]), dim=1)

    def _get_pred_all(self, outputs: dict):
        return torch.cat([outputs["ex_pred"], outputs["qn_pred"]], dim=1)

    def _build_train_inputs(self, batch: PyTree):
        return batch["data"]

    def _loss_all(self, batch: PyTree) -> dict[str, torch.Tensor]:
        data = self._build_train_inputs(batch)
        pred_icon = self.net.predict_backbone(data=data, mode="train", need_weights=False)
        hidden = self.net.extract_hidden(data=data, mode="train")

        train_label = self._get_ground_truth_all(batch)
        loss_icon = (self._get_pred_all(pred_icon) - train_label).pow(2).flatten(1).mean(dim=1)

        qn_target = batch["label"][:, 0]
        qn_baseline = pred_icon["qn_pred"][:, 0]
        loss_fm = self.net.compute_flow_matching_loss(target=qn_target, cond=hidden, baseline=qn_baseline)

        loss = self.loss_icon_weight * loss_icon + self.loss_fm_weight * loss_fm
        return {
            "loss": loss,
            "loss_icon": loss_icon,
            "loss_fm": loss_fm,
        }

    def get_preds(self, data: PyTree) -> dict[str, torch.Tensor]:
        pred_icon = self.net.predict_backbone(data=data, mode="test", need_weights=False)
        hidden = self.net.extract_hidden(data=data, mode="test")
        pred_dicon = self.net.sample_from_flow(
            cond=hidden,
            baseline=pred_icon["qn_pred"],
            num_steps=self.eval_flow_steps,
            num_samples=self.eval_num_samples,
            target_size=data["qn_f"].shape[-2:],
        )
        return {
            "quest_qoi_v_icon": pred_icon["qn_pred"],
            "quest_qoi_v_dicon": pred_dicon[:, None],
            "hidden": hidden,
        }

    def get_errors(self, preds: dict[str, torch.Tensor], batch: PyTree) -> dict[str, torch.Tensor]:
        return {
            "quest_qoi_v_icon": torch.abs(preds["quest_qoi_v_icon"] - batch["label"]).flatten(1).mean(dim=1),
            "quest_qoi_v_dicon": torch.abs(preds["quest_qoi_v_dicon"] - batch["label"]).flatten(1).mean(dim=1),
            "quest_qoi_v_gain": torch.abs(preds["quest_qoi_v_icon"] - batch["label"]).flatten(1).mean(dim=1)
            - torch.abs(preds["quest_qoi_v_dicon"] - batch["label"]).flatten(1).mean(dim=1),
        }

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

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> dict:
        losses = self._loss_all(batch)
        preds = self.get_preds(batch["data"])
        errors = self.get_errors(preds, batch)

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