import torch
from omegaconf import DictConfig
from optree import PyTree
from torchmetrics import MeanMetric, MetricCollection

import src.utils.icon_core_utils as cu
from src.plmodules.base_lit_module import BaseLitModule


class DiconLitModule(BaseLitModule):
    """Lightning module for D-ICON with baseline ICON comparison metrics."""

    def __init__(
        self,
        cfg: DictConfig,
        loss_icon_weight: float,
        loss_fm_weight: float,
        detach_cond_for_fm: bool,
        eval_flow_steps: int,
        eval_num_samples: int,
    ) -> None:
        super().__init__(cfg)
        self._net_compiled = True  # keep parity with ICON module behavior
        self.loss_icon_weight = loss_icon_weight
        self.loss_fm_weight = loss_fm_weight
        self.detach_cond_for_fm = detach_cond_for_fm
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

    def _build_train_label(self, batch: PyTree) -> torch.Tensor:
        demo_qoi_v = batch["data"]["demo_qoi_v"][:, self.cfg.loss.shot_num_min :, :, :]
        train_label = torch.cat([demo_qoi_v, batch["label"]], dim=1)
        return train_label

    def _get_backbone_pred_train(self, data: PyTree) -> torch.Tensor:
        pred = self.net.predict_backbone(data=data, mode="train", need_weights=False)
        return pred

    def _get_backbone_pred_test(self, data: PyTree) -> tuple[torch.Tensor, torch.Tensor]:
        pred, attn_weights = self.net.predict_backbone(data=data, mode="test", need_weights=True)
        return pred, attn_weights

    def _loss_function(self, batch: PyTree) -> dict[str, torch.Tensor]:
        pred_icon = self._get_backbone_pred_train(batch["data"])
        train_label = self._build_train_label(batch)

        loss_icon = (pred_icon - train_label).pow(2).flatten(1).mean(dim=1)

        cond = pred_icon.detach() if self.detach_cond_for_fm else pred_icon
        loss_fm = self.net.compute_flow_matching_loss(target=train_label, cond=cond)

        loss = self.loss_icon_weight * loss_icon + self.loss_fm_weight * loss_fm
        return {
            "loss": loss,
            "loss_icon": loss_icon,
            "loss_fm": loss_fm,
        }

    def get_preds(self, data: PyTree) -> dict[str, torch.Tensor]:
        """Get ICON baseline and D-ICON sampled predictions."""
        pred_icon, attn_weights = self._get_backbone_pred_test(data)
        pred_dicon = self.net.sample_from_flow(
            cond=pred_icon,
            num_steps=self.eval_flow_steps,
            num_samples=self.eval_num_samples,
        )
        return {
            "quest_qoi_v_icon": pred_icon,
            "quest_qoi_v_dicon": pred_dicon,
            "attn_weights": attn_weights,
        }

    def get_errors(self, preds: dict[str, torch.Tensor], batch: PyTree) -> dict[str, torch.Tensor]:
        """Compute validation errors for baseline ICON and D-ICON."""
        error_icon = torch.abs(preds["quest_qoi_v_icon"] - batch["label"]).flatten(1).mean(dim=1)
        error_dicon = torch.abs(preds["quest_qoi_v_dicon"] - batch["label"]).flatten(1).mean(dim=1)
        return {
            "quest_qoi_v_icon": error_icon,
            "quest_qoi_v_dicon": error_dicon,
            "quest_qoi_v_gain": error_icon - error_dicon,
        }

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics.values():
            metrics.reset()

    def training_step(self, batch: PyTree, batch_idx: int) -> torch.Tensor:
        losses = self._loss_function(batch)

        self.train_metrics(losses["loss"])
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=True)
        self.log("train/loss_icon", losses["loss_icon"].mean(), on_step=True, on_epoch=True)
        self.log("train/loss_fm", losses["loss_fm"].mean(), on_step=True, on_epoch=True)

        return losses["loss"].mean()

    def validation_step(self, batch: PyTree, batch_idx: int, dataloader_idx: int = 0) -> dict:
        losses = self._loss_function(batch)
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
        return {
            "preds": preds,
            "errors": errors,
            "metrics": metrics,
        }
