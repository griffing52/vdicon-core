from __future__ import annotations

import argparse
import csv
import gc
import importlib
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROJECT_ROOT", str(repo_root))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.datasets.pdearena.pdearena_incomp_split import PDEArenaIncompSplitDataset


KNOWN_MODEL_TARGETS = {
    # "LTX-VICON v1": "src.models.vicon.ltx_vicon.LTXViconVideoModel",
    "LTX-VICON": "src.models.vicon.ltx_vicon_v2.LTXViconVideoModelV2",
    "VICON transformer video": "src.models.vicon.vicon_transformer_video.ViconTransformerVideoModel",
}


CURATED_BASELINES = [
    ("Persistence", "", "persistence"),
    ("Pretrained VICON", "", "pretrained_vicon"),
    ("VICON", "logs/train/runs/VICON_100000_INCOMP/checkpoints/step_100000.ckpt", "lightning"),
    ("DICON", "logs/train/runs/DICON_100000_INCOMP/checkpoints/step_100000.ckpt", "lightning"),
    ("Latent VDICON", "logs/train/runs/LATENT_DICON_100000_INCOMP/checkpoints/step_100000.ckpt", "lightning"),
    ("VDICON v2", "logs/train/runs/VDICON_V2_100000_INCOMP/checkpoints/step_100000.ckpt", "lightning"),
    ("Pretrained VICON flow v2", "logs/train/runs/2026-06-09_07-21-52-572951/checkpoints/step_100000.ckpt", "lightning"),
    ("VICON transformer video", "logs/train/runs/VICON_TRANSFORMER_VIDEO/checkpoints/step_100000.ckpt", "checkpoint"),
    ("LTX-VICON", "logs/train/runs/LTX_VICON_V2/checkpoints/step_100000.ckpt", "checkpoint"),
]


@dataclass
class ModelSpec:
    name: str
    checkpoint_path: Path | None
    kind: str = "checkpoint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper tables and figures for VICON/PDEArena experiments.")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "paper_artifacts")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-auto-models", action="store_true")
    parser.add_argument("--skip-curated-baselines", action="store_true")
    parser.add_argument("--include-pretrained-vicon", action="store_true")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="Explicit raw-network model checkpoint to include. Can be repeated.",
    )
    parser.add_argument(
        "--lightning-model",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="Explicit LightningModule checkpoint to include. Can be repeated.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)\.ckpt$", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint_in_run(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "checkpoints").glob("step_*.ckpt"), key=checkpoint_step)
    if candidates:
        return candidates[-1]
    last = run_dir / "checkpoints" / "last.ckpt"
    return last if last.exists() else None


def run_metadata(run_dir: Path) -> str:
    paths = [run_dir / ".hydra" / "config.yaml", run_dir / "csv" / "version_0" / "hparams.yaml", run_dir / "tags.log"]
    return "\n".join(path.read_text(errors="ignore") for path in paths if path.exists())


def checkpoint_is_compatible(checkpoint_path: Path, target: str) -> bool:
    cfg_path = checkpoint_path.parents[1] / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return False
    cfg = OmegaConf.load(cfg_path)
    if str(cfg.model._target_) != target:
        return False
    if target.endswith("LTXViconVideoModelV2") and cfg.model.get("output_revision") != "target_stats_residual":
        return False
    return True


def auto_discover_models() -> list[ModelSpec]:
    run_root = repo_root / "logs" / "train" / "runs"
    specs: list[ModelSpec] = []
    for name, target in KNOWN_MODEL_TARGETS.items():
        candidates = []
        for run_dir in run_root.glob("*"):
            if not run_dir.is_dir() or target not in run_metadata(run_dir):
                continue
            checkpoint = latest_checkpoint_in_run(run_dir)
            if checkpoint is not None and checkpoint_is_compatible(checkpoint, target):
                candidates.append(checkpoint)
        if candidates:
            checkpoint = sorted(candidates, key=lambda path: (checkpoint_step(path), path.stat().st_mtime), reverse=True)[0]
            specs.append(ModelSpec(name=name, checkpoint_path=checkpoint))
    return specs


def parse_model_specs(raw_specs: list[str], kind: str = "checkpoint") -> list[ModelSpec]:
    specs = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"Expected NAME=CHECKPOINT, got {raw!r}")
        name, checkpoint = raw.split("=", 1)
        specs.append(ModelSpec(name=name.strip(), checkpoint_path=Path(checkpoint).expanduser().resolve(), kind=kind))
    return specs


def curated_baselines() -> list[ModelSpec]:
    specs = []
    for name, rel_checkpoint, kind in CURATED_BASELINES:
        if kind in {"pretrained_vicon", "persistence"}:
            specs.append(ModelSpec(name=name, checkpoint_path=None, kind=kind))
            continue
        checkpoint = repo_root / rel_checkpoint
        if checkpoint.exists():
            specs.append(ModelSpec(name=name, checkpoint_path=checkpoint, kind=kind))
    return specs



def is_persistence(spec_or_name: ModelSpec | str) -> bool:
    if isinstance(spec_or_name, ModelSpec):
        return spec_or_name.kind == "persistence" or spec_or_name.name.lower() == "persistence"
    return spec_or_name.lower() == "persistence"


def frame_mae(frame: torch.Tensor, target: torch.Tensor) -> float:
    return float((frame.detach().float().cpu() - target.detach().float().cpu()).abs().mean())


def order_frames_by_mae(frames: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    frame_map = dict(frames)
    gt = frame_map.get("Ground truth")
    fixed = [(name, frame) for name, frame in frames if name in ("Input", "Ground truth")]
    model_frames = [(name, frame) for name, frame in frames if name not in ("Input", "Ground truth") and name.lower() != "persistence"]
    if gt is not None:
        model_frames = sorted(model_frames, key=lambda item: frame_mae(item[1], gt))
    return fixed + model_frames


def rollout_row_mae(row: tuple[str, list[torch.Tensor]], gt_frames: list[torch.Tensor]) -> float:
    _, frames = row
    if not frames or not gt_frames:
        return float("inf")
    count = min(len(frames), len(gt_frames))
    return sum(frame_mae(frames[idx], gt_frames[idx]) for idx in range(count)) / count


def order_rollout_rows_by_mae(rollout_rows: list[tuple[str, list[torch.Tensor]]]) -> list[tuple[str, list[torch.Tensor]]]:
    if not rollout_rows:
        return rollout_rows
    gt_rows = [row for row in rollout_rows if row[0] == "Ground truth"]
    if not gt_rows:
        return [row for row in rollout_rows if row[0].lower() != "persistence"]
    gt = gt_rows[0]
    model_rows = [row for row in rollout_rows if row[0] != "Ground truth" and row[0].lower() != "persistence"]
    model_rows = sorted(model_rows, key=lambda row: rollout_row_mae(row, gt[1]))
    return [gt] + model_rows


def _gallery_feature(dataset: PDEArenaIncompSplitDataset, sample_idx: int, rollout_steps: int) -> torch.Tensor:
    qn_traj_idx, start_time = dataset._samples[sample_idx]
    ref = dataset._split_refs[qn_traj_idx]
    max_steps = (dataset._time_steps - 1 - start_time) // dataset.target_time_offset
    if max_steps < rollout_steps:
        raise ValueError("sample does not support requested rollout length")
    features = []
    for step in range(1, rollout_steps + 1):
        time_idx = start_time + step * dataset.target_time_offset
        frame = dataset._read_velocity_frame(ref, time_idx)
        speed = field_maps(frame)[0].unsqueeze(0).unsqueeze(0)
        pooled = torch.nn.functional.avg_pool2d(speed, kernel_size=16, stride=16).flatten()
        features.append(pooled)
    return torch.cat(features).float()


def select_diverse_gallery_indices(dataset: PDEArenaIncompSplitDataset, count: int, rollout_steps: int) -> list[int]:
    if len(dataset) <= count:
        return list(range(len(dataset)))
    candidate_count = min(240, len(dataset))
    if candidate_count == len(dataset):
        candidate_indices = list(range(len(dataset)))
    else:
        candidate_indices = sorted({round(i * (len(dataset) - 1) / (candidate_count - 1)) for i in range(candidate_count)})

    feature_pairs = []
    for idx in candidate_indices:
        try:
            traj_idx, _ = dataset._samples[idx]
            feature_pairs.append((idx, traj_idx, _gallery_feature(dataset, idx, rollout_steps)))
        except Exception:
            continue
    if len(feature_pairs) <= count:
        return [idx for idx, _, _ in feature_pairs]

    features = torch.stack([feature for _, _, feature in feature_pairs])
    trajs = [traj_idx for _, traj_idx, _ in feature_pairs]
    enough_unique_trajs = len(set(trajs)) >= count
    mean_feature = features.mean(dim=0, keepdim=True)
    distances_from_mean = (features - mean_feature).abs().mean(dim=1)
    selected_positions = [int(distances_from_mean.argmax())]

    while len(selected_positions) < count:
        selected_features = features[selected_positions]
        pairwise = (features[:, None, :] - selected_features[None, :, :]).abs().mean(dim=2)
        min_distance = pairwise.min(dim=1).values
        min_distance[selected_positions] = -1.0
        if enough_unique_trajs:
            selected_trajs = {trajs[pos] for pos in selected_positions}
            for pos, traj_idx in enumerate(trajs):
                if traj_idx in selected_trajs:
                    min_distance[pos] = -1.0
        selected_positions.append(int(min_distance.argmax()))

    return [feature_pairs[pos][0] for pos in selected_positions]


def plot_pretrained_backbone_rollout_gallery(
    path: Path,
    rows_by_sample: list[tuple[int, list[tuple[str, list[torch.Tensor]]]]],
    title: str,
) -> None:
    if not rows_by_sample:
        return
    model_names = ["Ground truth", "Pretrained VICON", "Pretrained VICON flow v2", "VICON transformer video"]
    steps = min(len(row[1]) for _, rows in rows_by_sample for row in rows)
    if steps <= 0:
        return
    total_rows = len(rows_by_sample) * len(model_names)
    stack = []
    for _, rows in rows_by_sample:
        row_map = dict(rows)
        for name in model_names:
            if name in row_map:
                stack.extend(field_maps(frame)[0] for frame in row_map[name][:steps])
    if not stack:
        return
    vmax = float(torch.stack(stack).max())
    fig, axes = plt.subplots(total_rows, steps, figsize=(2.45 * steps, 1.65 * total_rows), constrained_layout=True)
    if total_rows == 1:
        axes = axes[None, :]
    if steps == 1:
        axes = axes[:, None]
    out_row = 0
    for sample_idx, rows in rows_by_sample:
        row_map = dict(rows)
        for name in model_names:
            if name not in row_map:
                continue
            for col_idx, frame in enumerate(row_map[name][:steps]):
                ax = axes[out_row, col_idx]
                handle = ax.imshow(field_maps(frame)[0].numpy(), origin="lower", vmin=0.0, vmax=vmax, cmap="viridis")
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(f"s{sample_idx}\n{name}", fontsize=8)
                if out_row == 0:
                    ax.set_title(f"t+{col_idx + 1}")
            out_row += 1
    fig.colorbar(handle, ax=axes.ravel().tolist(), fraction=0.014, pad=0.01)
    fig.suptitle(title)
    fig.savefig(path, dpi=230)
    plt.close(fig)

def dedupe_specs(specs: list[ModelSpec]) -> list[ModelSpec]:
    seen_names = set()
    deduped = []
    for spec in specs:
        key = slug(spec.name).lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        deduped.append(spec)
    return deduped


def move_to_device(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def import_object(path: str):
    module_name, object_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), object_name)


def load_checkpoint_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    assert spec.checkpoint_path is not None
    checkpoint_path = spec.checkpoint_path.expanduser().resolve()
    config_path = checkpoint_path.parents[1] / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    cfg.paths.root_dir = str(repo_root)
    cfg.paths.data_dir = str(repo_root / "data")

    if str(cfg.model._target_).endswith("LTXViconVideoModelV2") and cfg.model.get("output_revision") != "target_stats_residual":
        raise ValueError(f"Skipping incompatible pre-fix LTX-VICON v2 checkpoint: {checkpoint_path}")

    model = hydra.utils.instantiate(cfg.model).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    net_state = {key.removeprefix("net."): value for key, value in state_dict.items() if key.startswith("net.")}
    if not net_state:
        net_state = state_dict
    model.load_state_dict(net_state, strict=True)
    return model


def load_lightning_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    assert spec.checkpoint_path is not None
    checkpoint_path = spec.checkpoint_path.expanduser().resolve()
    config_path = checkpoint_path.parents[1] / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    plmodule_cls = import_object(str(cfg.plmodule._target_))
    model = plmodule_cls.load_from_checkpoint(checkpoint_path, map_location=device, weights_only=False)
    return model.to(device).eval()


def load_pretrained_vicon(device: torch.device) -> torch.nn.Module:
    cfg = OmegaConf.create(
        {
            "paths": {"data_dir": str(repo_root / "data")},
            "model": OmegaConf.load(repo_root / "configs" / "model" / "vicon_dicon_latent_pretrained.yaml"),
        }
    )
    return hydra.utils.instantiate(cfg.model).to(device).eval()


def pick_prediction(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, torch.Tensor):
        return output
    for key in ("quest_qoi_v_dicon", "quest_qoi_v", "qn_pred", "quest_qoi_v_icon"):
        if isinstance(output, dict) and key in output:
            return output[key]
    raise KeyError(f"Could not find a prediction tensor in output keys: {list(output) if isinstance(output, dict) else type(output)}")


def predict_persistence(data: dict[str, torch.Tensor]) -> torch.Tensor:
    return data["qn_f"]


def predict(model: torch.nn.Module, data: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        if hasattr(model, "get_preds"):
            output = model.get_preds(data)
        elif hasattr(model, "network_inference"):
            output = model.network_inference(data)
        elif hasattr(model, "predict_backbone"):
            output = model.predict_backbone(data, mode="test", need_weights=False)
        else:
            output = model(data=data, mode="test")
    pred = pick_prediction(output)
    if pred.ndim == 4:
        pred = pred[:, None]
    return pred


def to_frame(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    while tensor.ndim > 3:
        tensor = tensor[0]
    return tensor


def field_maps(frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if frame.shape[0] == 7:
        u_comp, v_comp = frame[1], frame[2]
    elif frame.shape[0] == 2:
        u_comp, v_comp = frame[0], frame[1]
    else:
        raise ValueError(f"Expected 2 or 7 channels, got {frame.shape[0]}")
    speed = torch.linalg.norm(torch.stack([u_comp, v_comp], dim=0), dim=0)
    return speed, u_comp, v_comp


def scalar_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    diff = pred - target
    mae = diff.abs().mean()
    rmse = torch.sqrt(diff.square().mean())
    rel_rmse = rmse / (torch.sqrt(target.square().mean()) + 1e-8)

    pred_frame = to_frame(pred)
    target_frame = to_frame(target)
    pred_speed, pred_u, pred_v = field_maps(pred_frame)
    gt_speed, gt_u, gt_v = field_maps(target_frame)

    def mae_component(a, b):
        return (a - b).abs().mean()

    def rrmse_component(a, b):
        return torch.sqrt((a - b).square().mean()) / (torch.sqrt(b.square().mean()) + 1e-8)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "rrmse": float(rel_rmse),
        "speed_mae": float(mae_component(pred_speed, gt_speed)),
        "u_mae": float(mae_component(pred_u, gt_u)),
        "v_mae": float(mae_component(pred_v, gt_v)),
        "speed_rrmse": float(rrmse_component(pred_speed, gt_speed)),
        "u_rrmse": float(rrmse_component(pred_u, gt_u)),
        "v_rrmse": float(rrmse_component(pred_v, gt_v)),
    }


def summarize_metric_rows(rows: list[dict[str, float]], model_name: str) -> dict[str, float | str]:
    keys = [key for key, value in rows[0].items() if key not in ("sample_idx", "model") and isinstance(value, (int, float))]
    summary: dict[str, float | str] = {"model": model_name}
    for key in keys:
        values = torch.tensor([row[key] for row in rows], dtype=torch.float32)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(unbiased=False))
    return summary


def clean_generated_outputs(output_dir: Path) -> None:
    generated_names = {
        "model_specs.csv",
        "sample_metrics.csv",
        "rollout_metrics.csv",
        "metrics_summary.csv",
        "metrics_summary.md",
        "metrics_summary.tex",
        "qualitative_single_step.png",
        "qualitative_error_maps.png",
        "qualitative_improvement_vs_vicon.png",
        "rollout_speed.png",
        "rollout_u.png",
        "rollout_v.png",
        "metric_bar_mae.png",
        "metric_bar_rrmse.png",
        "metric_bar_components.png",
        "sample_mae_boxplot.png",
        "rollout_mae_curves.png",
        "ablation_runs.csv",
        "ablation_runs.md",
        "ablation_runs.tex",
        "ablation_valid_mae.png",
        "ablation_loss_vs_mae.png",
        "architecture_diagram.png",
    }
    for path in output_dir.glob("rollout_metrics_*.csv"):
        path.unlink()
    for name in generated_names:
        path = output_dir / name
        if path.exists():
            path.unlink()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        path.write_text("No rows.\n")
        return
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.5f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_latex_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        path.write_text("% No rows.\n")
        return
    latex_newline = r"\\"
    lines = ["\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}", "\\toprule"]
    lines.append(" & ".join(columns).replace("_", "\\_") + f" {latex_newline}")
    lines.append("\\midrule")
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.5f}" if isinstance(value, float) else str(value).replace("_", "\\_"))
        lines.append(" & ".join(values) + f" {latex_newline}")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")

def plot_component_grid(path: Path, frames: list[tuple[str, torch.Tensor]], title: str) -> None:
    speed_stack = torch.stack([field_maps(frame)[0] for _, frame in frames])
    component_stack = torch.stack([field_maps(frame)[idx] for _, frame in frames for idx in (1, 2)])
    speed_vmax = float(speed_stack.max())
    component_vmax = float(component_stack.abs().max())

    fig, axes = plt.subplots(3, len(frames), figsize=(3.3 * len(frames), 9.5), constrained_layout=True)
    if len(frames) == 1:
        axes = axes[:, None]
    for col, (name, frame) in enumerate(frames):
        speed, u_comp, v_comp = field_maps(frame)
        panels = [
            (speed, "speed", 0.0, speed_vmax, "viridis"),
            (u_comp, "u", -component_vmax, component_vmax, "coolwarm"),
            (v_comp, "v", -component_vmax, component_vmax, "coolwarm"),
        ]
        for row, (image, label, vmin, vmax, cmap) in enumerate(panels):
            ax = axes[row, col]
            handle = ax.imshow(image.numpy(), origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{name} {label}")
    fig.colorbar(handle, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_rollout_grid(path: Path, rollout_rows: list[tuple[str, list[torch.Tensor]]], title: str) -> None:
    if not rollout_rows or not rollout_rows[0][1]:
        return
    steps = len(rollout_rows[0][1])
    components = [(0, "speed", "viridis", False), (1, "u", "coolwarm", True), (2, "v", "coolwarm", True)]
    for component_idx, component_name, cmap, symmetric in components:
        component_rows = [(name, [field_maps(frame)[component_idx] for frame in frames]) for name, frames in rollout_rows]
        stack = torch.stack([image for _, row in component_rows for image in row])
        if symmetric:
            vmax = float(stack.abs().max())
            vmin = -vmax
        else:
            vmin = 0.0
            vmax = float(stack.max())
        fig, axes = plt.subplots(len(component_rows), steps, figsize=(2.7 * steps, 2.3 * len(component_rows)), constrained_layout=True)
        if steps == 1:
            axes = axes[:, None]
        for row_idx, (row_name, images) in enumerate(component_rows):
            for col_idx, image in enumerate(images):
                ax = axes[row_idx, col_idx]
                handle = ax.imshow(image.numpy(), origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(row_name)
                if row_idx == 0:
                    ax.set_title(f"t+{col_idx + 1}")
        fig.colorbar(handle, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
        fig.suptitle(f"{title}: {component_name}")
        fig.savefig(path.with_name(f"{path.stem}_{component_name}{path.suffix}"), dpi=220)
        plt.close(fig)



def plot_error_grid(path: Path, frames: list[tuple[str, torch.Tensor]], title: str) -> None:
    gt = dict(frames).get("Ground truth")
    if gt is None:
        return
    model_frames = [(name, frame) for name, frame in frames if name not in ("Input", "Ground truth")]
    if not model_frames:
        return
    error_maps = []
    for name, frame in model_frames:
        pred_fields = field_maps(frame)
        gt_fields = field_maps(gt)
        error_maps.append((name, [(pred_fields[idx] - gt_fields[idx]).abs() for idx in range(3)]))
    stack = torch.stack([image for _, row in error_maps for image in row])
    vmax = float(stack.max())
    labels = ["speed error", "u error", "v error"]
    fig, axes = plt.subplots(3, len(error_maps), figsize=(3.1 * len(error_maps), 8.8), constrained_layout=True)
    if len(error_maps) == 1:
        axes = axes[:, None]
    for col, (name, images) in enumerate(error_maps):
        for row, image in enumerate(images):
            ax = axes[row, col]
            handle = ax.imshow(image.numpy(), origin="lower", vmin=0.0, vmax=vmax, cmap="magma")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{name} {labels[row]}")
    fig.colorbar(handle, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_improvement_grid(path: Path, frames: list[tuple[str, torch.Tensor]], baseline_name: str = "VICON") -> None:
    frame_map = dict(frames)
    gt = frame_map.get("Ground truth")
    baseline = frame_map.get(baseline_name)
    if gt is None or baseline is None:
        return
    targets = [(name, frame) for name, frame in frames if name not in ("Input", "Ground truth", baseline_name)]
    if not targets:
        return
    gt_fields = field_maps(gt)
    baseline_fields = field_maps(baseline)
    baseline_errors = [(baseline_fields[idx] - gt_fields[idx]).abs() for idx in range(3)]
    rows = []
    for name, frame in targets:
        pred_fields = field_maps(frame)
        rows.append((name, [baseline_errors[idx] - (pred_fields[idx] - gt_fields[idx]).abs() for idx in range(3)]))
    stack = torch.stack([image for _, row in rows for image in row])
    vmax = float(stack.abs().max())
    labels = ["speed gain", "u gain", "v gain"]
    fig, axes = plt.subplots(3, len(rows), figsize=(3.1 * len(rows), 8.8), constrained_layout=True)
    if len(rows) == 1:
        axes = axes[:, None]
    for col, (name, images) in enumerate(rows):
        for row_idx, image in enumerate(images):
            ax = axes[row_idx, col]
            handle = ax.imshow(image.numpy(), origin="lower", vmin=-vmax, vmax=vmax, cmap="PiYG")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{name} {labels[row_idx]}")
    fig.colorbar(handle, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    fig.suptitle(f"Error improvement over {baseline_name} (positive is better)")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_metric_bar(path: Path, summary_rows: list[dict[str, Any]], metric: str, ylabel: str) -> None:
    if not summary_rows:
        return
    names = [str(row["model"]) for row in summary_rows]
    means = [float(row[f"{metric}_mean"]) for row in summary_rows]
    stds = [float(row.get(f"{metric}_std", 0.0)) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(max(7.5, 1.15 * len(names)), 4.6), constrained_layout=True)
    ax.bar(range(len(names)), means, yerr=stds, color="#5975a4", capsize=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_component_bars(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    names = [str(row["model"]) for row in summary_rows]
    components = [("speed_mae_mean", "speed"), ("u_mae_mean", "u"), ("v_mae_mean", "v")]
    width = 0.24
    x = torch.arange(len(names), dtype=torch.float32)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(names)), 4.8), constrained_layout=True)
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, (key, label) in enumerate(components):
        values = [float(row[key]) for row in summary_rows]
        ax.bar((x + (idx - 1) * width).tolist(), values, width=width, label=label, color=colors[idx])
    ax.set_xticks(x.tolist())
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_sample_boxplot(path: Path, sample_rows: list[dict[str, Any]], metric: str = "mae") -> None:
    if not sample_rows:
        return
    names = []
    grouped = []
    for row in sample_rows:
        name = str(row["model"])
        if name not in names:
            names.append(name)
            grouped.append([])
        grouped[names.index(name)].append(float(row[metric]))
    fig, ax = plt.subplots(figsize=(max(7.5, 1.15 * len(names)), 4.8), constrained_layout=True)
    ax.boxplot(grouped, tick_labels=names, showfliers=False)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(metric.upper())
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_rollout_metric_curves(path: Path, rollout_rows: list[dict[str, Any]], metric: str = "mae") -> None:
    if not rollout_rows:
        return
    names = []
    for row in rollout_rows:
        name = str(row["model"])
        if name not in names:
            names.append(name)
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    for name in names:
        rows = sorted([row for row in rollout_rows if row["model"] == name], key=lambda row: int(row["step"]))
        ax.plot([int(row["step"]) for row in rows], [float(row[metric]) for row in rows], marker="o", label=name)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel(metric.upper())
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_ablation_bars(path: Path, rows: list[dict[str, Any]], metric: str = "valid_mae", limit: int = 14) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row[metric]))[:limit]
    labels = [str(row["target"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.85 * len(labels)), 4.8), constrained_layout=True)
    ax.bar(range(len(labels)), values, color="#7f9f7a")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_ablation_scatter(path: Path, rows: list[dict[str, Any]]) -> None:
    filtered = [row for row in rows if not math.isnan(float(row.get("valid_loss", "nan")))]
    if not filtered:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    ax.scatter([float(row["valid_loss"]) for row in filtered], [float(row["valid_mae"]) for row in filtered], color="#b279a2")
    for row in filtered[:12]:
        ax.annotate(str(row["target"]), (float(row["valid_loss"]), float(row["valid_mae"])), fontsize=7, alpha=0.75)
    ax.set_xlabel("validation loss")
    ax.set_ylabel("validation MAE")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_architecture_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.axis("off")
    boxes = [
        (0.05, 0.55, "PDEArena prompt\nexamples + query"),
        (0.28, 0.75, "VICON-style\nchannel/patch tokens"),
        (0.52, 0.75, "Pretrained VICON\ntransformer core"),
        (0.76, 0.75, "Dense decoder\n512x512 velocity"),
        (0.28, 0.25, "LTX query/context\nadapters"),
        (0.52, 0.25, "LTX video\ntransformer core"),
        (0.76, 0.25, "Residual decoder\nqn_f + delta"),
    ]
    for x, y, text in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.45", fc="#f7f7f7", ec="#333333"), transform=ax.transAxes)
    arrows = [((0.13, 0.55), (0.22, 0.75)), ((0.36, 0.75), (0.44, 0.75)), ((0.60, 0.75), (0.68, 0.75)), ((0.13, 0.55), (0.22, 0.25)), ((0.36, 0.25), (0.44, 0.25)), ((0.60, 0.25), (0.68, 0.25))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.text(0.52, 0.94, "VICON-transformer baseline", ha="center", fontsize=13, weight="bold", transform=ax.transAxes)
    ax.text(0.52, 0.44, "LTX-VICON baseline", ha="center", fontsize=13, weight="bold", transform=ax.transAxes)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def collect_ablation_rows() -> list[dict[str, Any]]:
    rows = []
    for run_dir in (repo_root / "logs" / "train" / "runs").glob("*"):
        metrics_path = run_dir / "csv" / "version_0" / "metrics.csv"
        config_path = run_dir / ".hydra" / "config.yaml"
        if not metrics_path.exists() or not config_path.exists():
            continue
        cfg = OmegaConf.load(config_path)
        target = str(cfg.model._target_) if "model" in cfg and "_target_" in cfg.model else "unknown"
        with metrics_path.open() as handle:
            reader = csv.DictReader(handle)
            metric_rows = list(reader)
        valid_rows = [row for row in metric_rows if row.get("pdearena_incomp_valid/quest_qoi_v") not in (None, "", "nan", "NaN")]
        if not valid_rows:
            continue
        last = valid_rows[-1]
        rows.append(
            {
                "run": run_dir.name,
                "target": target.rsplit(".", 1)[-1],
                "step": int(float(last.get("step", 0))),
                "valid_mae": float(last["pdearena_incomp_valid/quest_qoi_v"]),
                "valid_loss": float(last.get("pdearena_incomp_valid/loss", "nan")),
            }
        )
    return sorted(rows, key=lambda row: row["valid_mae"])


def evaluate_models(specs: list[ModelSpec], dataset: PDEArenaIncompSplitDataset, args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    sample_indices = list(range(min(args.num_samples, len(dataset))))
    qualitative_idx = min(args.sample_idx, len(dataset) - 1)
    qualitative_sample = move_to_device(dataset[qualitative_idx], device)
    qualitative_frames = [("Input", to_frame(qualitative_sample["data"]["qn_f"])), ("Ground truth", to_frame(qualitative_sample["label"]))]

    all_sample_rows = []
    summary_rows = []
    all_rollout_metric_rows = []
    rollout_rows = [("Ground truth", [])]
    qn_traj_idx, rollout_start_time = dataset._samples[qualitative_idx]
    max_rollout_steps = (dataset._time_steps - 1 - rollout_start_time) // dataset.target_time_offset
    rollout_steps = max(1, min(args.rollout_steps, max_rollout_steps))
    rollout_ref = dataset._split_refs[qn_traj_idx]
    rollout_context_indices = dataset._sample_context_indices(qn_traj_idx, qualitative_idx)

    def read_rollout_frame(step: int) -> torch.Tensor:
        time_idx = rollout_start_time + step * dataset.target_time_offset
        return dataset._read_velocity_frame(rollout_ref, time_idx).unsqueeze(0).unsqueeze(0)

    def read_rollout_context(step: int, device: torch.device) -> dict[str, torch.Tensor]:
        time_idx = rollout_start_time + step * dataset.target_time_offset
        target_time_idx = time_idx + dataset.target_time_offset
        ex_f = torch.stack([dataset._read_velocity_frame(dataset._split_refs[idx], time_idx) for idx in rollout_context_indices]).unsqueeze(0).to(device)
        ex_g = torch.stack([dataset._read_velocity_frame(dataset._split_refs[idx], target_time_idx) for idx in rollout_context_indices]).unsqueeze(0).to(device)
        return {"ex_f": ex_f, "ex_g": ex_g}

    gt_rollout = [to_frame(read_rollout_frame(step)) for step in range(1, rollout_steps + 1)]
    rollout_rows[0] = ("Ground truth", gt_rollout)

    loaded_model_names: set[str] = set()
    pretrained_gallery_models: dict[str, torch.nn.Module | None] = {}

    for spec in specs:
        print(f"[paper] loading {spec.name}")
        if spec.kind == "persistence":
            model = None
        elif spec.kind == "pretrained_vicon":
            model = load_pretrained_vicon(device)
        elif spec.kind == "lightning":
            model = load_lightning_model(spec, device)
        else:
            model = load_checkpoint_model(spec, device)

        sample_metric_rows = []
        for sample_idx in sample_indices:
            sample = move_to_device(dataset[sample_idx], device)
            pred = predict_persistence(sample["data"]) if spec.kind == "persistence" else predict(model, sample["data"])
            metrics = scalar_metrics(pred, sample["label"])
            metrics.update({"model": spec.name, "sample_idx": sample_idx})
            sample_metric_rows.append(metrics)
            all_sample_rows.append(metrics)

        summary_rows.append(summarize_metric_rows(sample_metric_rows, spec.name))

        if not is_persistence(spec):
            pred = predict(model, qualitative_sample["data"])
            qualitative_frames.append((spec.name, to_frame(pred)))

        current = read_rollout_frame(0).to(device)
        model_rollout = []
        rollout_metric_rows = []
        for step in range(rollout_steps):
            step_data = {**read_rollout_context(step, device), "qn_f": current}
            pred = predict_persistence(step_data) if spec.kind == "persistence" else predict(model, step_data)
            frame = to_frame(pred)
            model_rollout.append(frame)
            rollout_metrics = scalar_metrics(pred, read_rollout_frame(step + 1).to(device))
            rollout_metrics.update({"model": spec.name, "step": step + 1})
            rollout_metric_rows.append(rollout_metrics)
            all_rollout_metric_rows.append(rollout_metrics)
            current = pred
        if not is_persistence(spec):
            rollout_rows.append((spec.name, model_rollout))
        write_csv(output_dir / f"rollout_metrics_{slug(spec.name)}.csv", rollout_metric_rows)

        if spec.name in {"Pretrained VICON", "Pretrained VICON flow v2", "VICON transformer video"} and model is not None:
            pretrained_gallery_models[spec.name] = model
            loaded_model_names.add(spec.name)
        elif model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    qualitative_frames = order_frames_by_mae(qualitative_frames)
    rollout_rows = order_rollout_rows_by_mae(rollout_rows)
    summary_rows = sorted(summary_rows, key=lambda row: float(row["mae_mean"]))
    visual_summary_rows = [row for row in summary_rows if str(row["model"]).lower() != "persistence"]
    visual_sample_rows = [row for row in all_sample_rows if str(row["model"]).lower() != "persistence"]
    visual_rollout_metric_rows = [row for row in all_rollout_metric_rows if str(row["model"]).lower() != "persistence"]

    write_csv(output_dir / "sample_metrics.csv", all_sample_rows)
    write_csv(output_dir / "rollout_metrics.csv", all_rollout_metric_rows)
    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    table_columns = ["model", "mae_mean", "rmse_mean", "rrmse_mean", "speed_mae_mean", "u_mae_mean", "v_mae_mean"]
    write_markdown_table(output_dir / "metrics_summary.md", summary_rows, table_columns)
    write_latex_table(output_dir / "metrics_summary.tex", summary_rows, table_columns)
    plot_component_grid(output_dir / "qualitative_single_step.png", qualitative_frames, title=f"Single-step predictions, sample {qualitative_idx}")
    plot_error_grid(output_dir / "qualitative_error_maps.png", qualitative_frames, title=f"Single-step absolute errors, sample {qualitative_idx}")
    plot_improvement_grid(output_dir / "qualitative_improvement_vs_vicon.png", qualitative_frames, baseline_name="VICON")
    plot_rollout_grid(output_dir / "rollout.png", rollout_rows, title=f"Autoregressive rollout, sample {qualitative_idx}")
    plot_metric_bar(output_dir / "metric_bar_mae.png", visual_summary_rows, "mae", "MAE")
    plot_metric_bar(output_dir / "metric_bar_rrmse.png", visual_summary_rows, "rrmse", "relative RMSE")
    plot_component_bars(output_dir / "metric_bar_components.png", visual_summary_rows)
    plot_sample_boxplot(output_dir / "sample_mae_boxplot.png", visual_sample_rows, metric="mae")
    plot_rollout_metric_curves(output_dir / "rollout_mae_curves.png", visual_rollout_metric_rows, metric="mae")

    if {"Pretrained VICON", "Pretrained VICON flow v2", "VICON transformer video"}.issubset(loaded_model_names):
        gallery = []
        gallery_dataset = dataset
        gallery_split = args.split
        unique_trajs = {traj_idx for traj_idx, _ in gallery_dataset._samples}
        if len(unique_trajs) < 3:
            gallery_split = "train"
            gallery_dataset = PDEArenaIncompSplitDataset(
                file_paths=str(repo_root / "data" / "pdearena_incomp" / "ns_incom_inhom_2d_512-*.h5"),
                ex_num=5,
                split=gallery_split,
            )
        gallery_indices = select_diverse_gallery_indices(gallery_dataset, count=3, rollout_steps=args.rollout_steps)
        gallery_meta = [gallery_dataset._samples[idx] for idx in gallery_indices]
        print(f"[paper] pretrained rollout gallery split={gallery_split} samples: {list(zip(gallery_indices, gallery_meta))}")

        for gallery_idx in gallery_indices:
            qn_traj_idx_i, start_time_i = gallery_dataset._samples[gallery_idx]
            max_steps_i = (gallery_dataset._time_steps - 1 - start_time_i) // gallery_dataset.target_time_offset
            steps_i = max(1, min(args.rollout_steps, max_steps_i))
            ref_i = gallery_dataset._split_refs[qn_traj_idx_i]
            context_indices_i = gallery_dataset._sample_context_indices(qn_traj_idx_i, gallery_idx)

            def read_sample_frame(step: int) -> torch.Tensor:
                time_idx = start_time_i + step * gallery_dataset.target_time_offset
                return gallery_dataset._read_velocity_frame(ref_i, time_idx).unsqueeze(0).unsqueeze(0)

            def read_sample_context(step: int) -> dict[str, torch.Tensor]:
                time_idx = start_time_i + step * gallery_dataset.target_time_offset
                target_time_idx = time_idx + gallery_dataset.target_time_offset
                ex_f = torch.stack([gallery_dataset._read_velocity_frame(gallery_dataset._split_refs[idx], time_idx) for idx in context_indices_i]).unsqueeze(0).to(device)
                ex_g = torch.stack([gallery_dataset._read_velocity_frame(gallery_dataset._split_refs[idx], target_time_idx) for idx in context_indices_i]).unsqueeze(0).to(device)
                return {"ex_f": ex_f, "ex_g": ex_g}

            rows = [("Ground truth", [to_frame(read_sample_frame(step)) for step in range(1, steps_i + 1)])]
            for name in ("Pretrained VICON", "Pretrained VICON flow v2", "VICON transformer video"):
                model = pretrained_gallery_models[name]
                current = read_sample_frame(0).to(device)
                states = []
                for step in range(steps_i):
                    pred = predict(model, {**read_sample_context(step), "qn_f": current})
                    states.append(to_frame(pred))
                    current = pred
                rows.append((name, states))
            gallery.append((gallery_idx, rows))
        plot_pretrained_backbone_rollout_gallery(
            output_dir / "pretrained_backbone_rollout_gallery.png",
            gallery,
            title="Pretrained-backbone rollouts on multiple PDEBench inputs",
        )

    for model in pretrained_gallery_models.values():
        if model is not None:
            del model


def slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_outputs(args.output_dir)
    dataset = PDEArenaIncompSplitDataset(
        file_paths=str(repo_root / "data" / "pdearena_incomp" / "ns_incom_inhom_2d_512-*.h5"),
        ex_num=5,
        split=args.split,
    )

    specs = []
    if not args.skip_curated_baselines:
        specs.extend(curated_baselines())
    if not args.skip_auto_models:
        specs.extend(auto_discover_models())
    specs.extend(parse_model_specs(args.model, kind="checkpoint"))
    specs.extend(parse_model_specs(args.lightning_model, kind="lightning"))
    if args.include_pretrained_vicon:
        specs.insert(0, ModelSpec(name="Pretrained VICON", checkpoint_path=None, kind="pretrained_vicon"))
    specs = dedupe_specs(specs)

    discovered_rows = [{"model": spec.name, "kind": spec.kind, "checkpoint": str(spec.checkpoint_path or "")} for spec in specs]
    write_csv(args.output_dir / "model_specs.csv", discovered_rows)
    print(f"[paper] models: {[spec.name for spec in specs]}")

    if specs:
        evaluate_models(specs, dataset, args)
    else:
        print("[paper] no models selected; skipping model evaluation")

    ablation_rows = collect_ablation_rows()
    write_csv(args.output_dir / "ablation_runs.csv", ablation_rows)
    write_markdown_table(args.output_dir / "ablation_runs.md", ablation_rows, ["run", "target", "step", "valid_mae", "valid_loss"])
    write_latex_table(args.output_dir / "ablation_runs.tex", ablation_rows, ["run", "target", "step", "valid_mae", "valid_loss"])
    plot_ablation_bars(args.output_dir / "ablation_valid_mae.png", ablation_rows, metric="valid_mae")
    plot_ablation_scatter(args.output_dir / "ablation_loss_vs_mae.png", ablation_rows)
    plot_architecture_diagram(args.output_dir / "architecture_diagram.png")
    print(f"[paper] wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
