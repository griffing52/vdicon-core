from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts_core.generate_paper_artifacts import (  # noqa: E402
    auto_discover_models,
    curated_baselines,
    dedupe_specs,
    field_maps,
    is_persistence,
    load_checkpoint_model,
    load_lightning_model,
    load_pretrained_vicon,
    move_to_device,
    predict,
    predict_persistence,
    slug,
    to_frame,
    write_csv,
)
from src.datasets.pdearena.pdearena_incomp_split import PDEArenaIncompSplitDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rollout update/delta figures for PDEArena model comparisons.")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "paper_artifacts" / "rollout_updates")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-auto-models", action="store_true")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model names to keep, e.g. VICON DICON 'VICON transformer video'.")
    return parser.parse_args()


def load_model(spec, device: torch.device):
    if spec.kind == "persistence":
        return None
    if spec.kind == "pretrained_vicon":
        return load_pretrained_vicon(device)
    if spec.kind == "lightning":
        return load_lightning_model(spec, device)
    return load_checkpoint_model(spec, device)


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu().flatten()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = flatten(a)
    bb = flatten(b)
    denom = aa.norm() * bb.norm() + 1e-12
    return float(torch.dot(aa, bb) / denom)


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().mean())


def update_norm(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu().abs().mean())


def component_image(frame: torch.Tensor, idx: int) -> torch.Tensor:
    return field_maps(frame)[idx]


def plot_grid(path: Path, rows: list[tuple[str, list[torch.Tensor]]], component_idx: int, title: str, symmetric: bool = True) -> None:
    if not rows or not rows[0][1]:
        return
    steps = len(rows[0][1])
    images = [(name, [component_image(frame, component_idx) for frame in frames]) for name, frames in rows]
    stack = torch.stack([image for _, row in images for image in row])
    if symmetric:
        vmax = float(stack.abs().max())
        vmin = -vmax
        cmap = "coolwarm"
    else:
        vmin = 0.0
        vmax = float(stack.max())
        cmap = "viridis"
    fig, axes = plt.subplots(len(images), steps, figsize=(2.45 * steps, 1.95 * len(images)), constrained_layout=True)
    if len(images) == 1:
        axes = axes[None, :]
    if steps == 1:
        axes = axes[:, None]
    for row_idx, (name, row_images) in enumerate(images):
        for col_idx, image in enumerate(row_images):
            ax = axes[row_idx, col_idx]
            handle = ax.imshow(image.numpy(), origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(name, fontsize=9)
            if row_idx == 0:
                ax.set_title(f"t+{col_idx + 1}")
    fig.colorbar(handle, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    fig.suptitle(title)
    fig.savefig(path, dpi=230)
    plt.close(fig)


def plot_update_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    names = []
    for row in rows:
        if row["model"] not in names:
            names.append(row["model"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for name in names:
        model_rows = sorted([row for row in rows if row["model"] == name], key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in model_rows]
        axes[0].plot(steps, [float(row["pred_update_mae"]) for row in model_rows], marker="o", label=name)
        axes[1].plot(steps, [float(row["update_cosine"]) for row in model_rows], marker="o", label=name)
    axes[0].set_title("Predicted update magnitude")
    axes[0].set_xlabel("Rollout step")
    axes[0].set_ylabel("MAE(pred_t - input_t)")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Update direction alignment")
    axes[1].set_xlabel("Rollout step")
    axes[1].set_ylabel("cosine(pred update, true update)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.savefig(path, dpi=230)
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("No rows.\n")
        return
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")



def row_mae(row: tuple[str, list[torch.Tensor]], gt_frames: list[torch.Tensor]) -> float:
    _, frames = row
    if not frames or not gt_frames:
        return float("inf")
    count = min(len(frames), len(gt_frames))
    return sum(mae(frames[idx], gt_frames[idx]) for idx in range(count)) / count


def order_rows_by_state_mae(rows: list[tuple[str, list[torch.Tensor]]]) -> list[tuple[str, list[torch.Tensor]]]:
    gt_rows = [row for row in rows if row[0] in ("Ground truth", "True update")]
    if not gt_rows:
        return [row for row in rows if row[0].lower() != "persistence"]
    gt = gt_rows[0]
    model_rows = [row for row in rows if row[0] != gt[0] and row[0].lower() != "persistence"]
    model_rows = sorted(model_rows, key=lambda row: row_mae(row, gt[1]))
    return [gt] + model_rows

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset = PDEArenaIncompSplitDataset(
        file_paths=str(repo_root / "data" / "pdearena_incomp" / "ns_incom_inhom_2d_512-*.h5"),
        ex_num=5,
        split=args.split,
    )
    qualitative_idx = min(args.sample_idx, len(dataset) - 1)
    sample = move_to_device(dataset[qualitative_idx], device)
    qn_traj_idx, rollout_start_time = dataset._samples[qualitative_idx]
    max_rollout_steps = (dataset._time_steps - 1 - rollout_start_time) // dataset.target_time_offset
    rollout_steps = max(1, min(args.rollout_steps, max_rollout_steps))
    rollout_ref = dataset._split_refs[qn_traj_idx]

    def read_frame(step: int) -> torch.Tensor:
        time_idx = rollout_start_time + step * dataset.target_time_offset
        return dataset._read_velocity_frame(rollout_ref, time_idx).unsqueeze(0).unsqueeze(0).to(device)

    specs = curated_baselines()
    if not args.skip_auto_models:
        specs.extend(auto_discover_models())
    specs = [spec for spec in dedupe_specs(specs) if not is_persistence(spec)]
    if args.models:
        keep = set(args.models)
        specs = [spec for spec in specs if spec.name in keep]

    model_rows: list[tuple[str, list[torch.Tensor]]] = []
    delta_rows: list[tuple[str, list[torch.Tensor]]] = []
    true_delta_frames = []
    true_state_frames = []
    metrics_rows = []

    for step in range(rollout_steps):
        prev_gt = read_frame(step)
        next_gt = read_frame(step + 1)
        true_state_frames.append(to_frame(next_gt))
        true_delta_frames.append(to_frame(next_gt - prev_gt))
    model_rows.append(("Ground truth", true_state_frames))
    delta_rows.append(("True update", true_delta_frames))

    context_indices = dataset._sample_context_indices(qn_traj_idx, qualitative_idx)

    def read_context(step: int) -> dict[str, torch.Tensor]:
        time_idx = rollout_start_time + step * dataset.target_time_offset
        target_time_idx = time_idx + dataset.target_time_offset
        ex_f = torch.stack([dataset._read_velocity_frame(dataset._split_refs[idx], time_idx) for idx in context_indices]).unsqueeze(0).to(device)
        ex_g = torch.stack([dataset._read_velocity_frame(dataset._split_refs[idx], target_time_idx) for idx in context_indices]).unsqueeze(0).to(device)
        return {"ex_f": ex_f, "ex_g": ex_g}

    for spec in specs:
        print(f"[rollout-updates] loading {spec.name}")
        model = load_model(spec, device)
        current = read_frame(0)
        states = []
        deltas = []
        for step in range(rollout_steps):
            step_data = {**read_context(step), "qn_f": current}
            pred = predict(model, step_data)
            gt_next = read_frame(step + 1)
            pred_delta = pred - current
            true_delta_from_model_state = gt_next - current
            states.append(to_frame(pred))
            deltas.append(to_frame(pred_delta))
            metrics_rows.append(
                {
                    "model": spec.name,
                    "step": step + 1,
                    "state_mae": mae(pred, gt_next),
                    "pred_update_mae": update_norm(pred_delta),
                    "true_update_from_model_state_mae": update_norm(true_delta_from_model_state),
                    "update_ratio": update_norm(pred_delta) / (update_norm(true_delta_from_model_state) + 1e-12),
                    "update_cosine": cosine(pred_delta, true_delta_from_model_state),
                }
            )
            current = pred
        model_rows.append((spec.name, states))
        delta_rows.append((spec.name, deltas))
        if model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model_rows = order_rows_by_state_mae(model_rows)
    delta_rows = order_rows_by_state_mae(delta_rows)
    metrics_rows = sorted(metrics_rows, key=lambda row: (float(row["state_mae"]), str(row["model"]), int(row["step"])))

    write_csv(args.output_dir / "rollout_update_metrics.csv", metrics_rows)
    write_markdown(args.output_dir / "rollout_update_metrics.md", metrics_rows)
    plot_grid(args.output_dir / "rollout_state_speed.png", model_rows, component_idx=0, title="Rollout states: speed", symmetric=False)
    plot_grid(args.output_dir / "rollout_update_speed.png", delta_rows, component_idx=0, title="Rollout updates: speed delta", symmetric=True)
    plot_grid(args.output_dir / "rollout_update_u.png", delta_rows, component_idx=1, title="Rollout updates: u delta", symmetric=True)
    plot_grid(args.output_dir / "rollout_update_v.png", delta_rows, component_idx=2, title="Rollout updates: v delta", symmetric=True)
    plot_update_lines(args.output_dir / "rollout_update_magnitude_alignment.png", metrics_rows)

    summary = [
        "# Rollout Update Artifacts",
        "",
        "These figures visualize `prediction - current input` during autoregressive rollout with dynamically advanced in-context examples.",
        "Rows exclude persistence and are ordered by rollout MAE so stronger models appear nearest to the ground truth.",
        "",
        f"Split: `{args.split}`; sample index: `{qualitative_idx}`; rollout steps: `{rollout_steps}`.",
        "",
        "Files:",
        "- `rollout_state_speed.png`: state rollout in speed space.",
        "- `rollout_update_speed.png`, `rollout_update_u.png`, `rollout_update_v.png`: predicted update maps by component.",
        "- `rollout_update_magnitude_alignment.png`: update magnitude and directional alignment over rollout steps.",
        "- `rollout_update_metrics.{csv,md}`: numeric update metrics.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(summary) + "\n")
    print(f"[rollout-updates] wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
