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
    parser = argparse.ArgumentParser(description="Compare static, dynamic, and teacher-forced rollout context modes.")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "paper_artifacts" / "rollout_context_diagnostics")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def load_model(spec, device: torch.device):
    if spec.kind == "persistence":
        return None
    if spec.kind == "pretrained_vicon":
        return load_pretrained_vicon(device)
    if spec.kind == "lightning":
        return load_lightning_model(spec, device)
    return load_checkpoint_model(spec, device)


def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(x.detach().float().cpu().square().mean()))


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().mean())


def tv(x: torch.Tensor) -> float:
    x = x.detach().float().cpu()
    while x.ndim > 4:
        x = x[0]
    return float((x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().float().cpu().flatten()
    bb = b.detach().float().cpu().flatten()
    return float(torch.dot(aa, bb) / (aa.norm() * bb.norm() + 1e-12))


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


def plot_lines(path: Path, rows: list[dict[str, Any]], metric: str, ylabel: str) -> None:
    keys = []
    for row in rows:
        key = (row["model"], row["mode"])
        if key not in keys:
            keys.append(key)
    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    for model, mode in keys:
        group = sorted([row for row in rows if row["model"] == model and row["mode"] == mode], key=lambda row: int(row["step"]))
        ax.plot([int(row["step"]) for row in group], [float(row[metric]) for row in group], marker="o", label=f"{model} / {mode}")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.savefig(path, dpi=230)
    plt.close(fig)


def plot_mode_grid(path: Path, rows: list[tuple[str, list[torch.Tensor]]], title: str) -> None:
    if not rows:
        return
    steps = len(rows[0][1])
    images = [(name, [field_maps(frame)[0] for frame in frames]) for name, frames in rows]
    stack = torch.stack([image for _, row in images for image in row])
    vmax = float(stack.max())
    fig, axes = plt.subplots(len(images), steps, figsize=(2.25 * steps, 1.75 * len(images)), constrained_layout=True)
    if len(images) == 1:
        axes = axes[None, :]
    if steps == 1:
        axes = axes[:, None]
    for row_idx, (name, row_images) in enumerate(images):
        for col_idx, image in enumerate(row_images):
            ax = axes[row_idx, col_idx]
            handle = ax.imshow(image.numpy(), origin="lower", vmin=0.0, vmax=vmax, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(name, fontsize=8)
            if row_idx == 0:
                ax.set_title(f"t+{col_idx + 1}")
    fig.colorbar(handle, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    fig.suptitle(title)
    fig.savefig(path, dpi=230)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = PDEArenaIncompSplitDataset(
        file_paths=str(repo_root / "data" / "pdearena_incomp" / "ns_incom_inhom_2d_512-*.h5"),
        ex_num=5,
        split=args.split,
    )
    sample_idx = min(args.sample_idx, len(dataset) - 1)
    base_sample = move_to_device(dataset[sample_idx], device)
    qn_traj_idx, start_time = dataset._samples[sample_idx]
    context_indices = dataset._sample_context_indices(qn_traj_idx, sample_idx)
    max_rollout_steps = (dataset._time_steps - 1 - start_time) // dataset.target_time_offset
    steps = max(1, min(args.rollout_steps, max_rollout_steps))
    qn_ref = dataset._split_refs[qn_traj_idx]

    def frame(ref, time_idx: int) -> torch.Tensor:
        return dataset._read_velocity_frame(ref, time_idx).unsqueeze(0).unsqueeze(0).to(device)

    def qn_frame(step: int) -> torch.Tensor:
        return frame(qn_ref, start_time + step * dataset.target_time_offset)

    def context_at(step: int) -> dict[str, torch.Tensor]:
        time_idx = start_time + step * dataset.target_time_offset
        target_time_idx = time_idx + dataset.target_time_offset
        ex_f = torch.stack([dataset._read_velocity_frame(dataset._split_refs[idx], time_idx) for idx in context_indices]).unsqueeze(0).to(device)
        ex_g = torch.stack([dataset._read_velocity_frame(dataset._split_refs[idx], target_time_idx) for idx in context_indices]).unsqueeze(0).to(device)
        return {"ex_f": ex_f, "ex_g": ex_g}

    specs = curated_baselines() + auto_discover_models()
    specs = dedupe_specs(specs)
    if args.models:
        keep = set(args.models)
        specs = [spec for spec in specs if spec.name in keep]

    modes = ["static_context", "dynamic_context", "teacher_forced"]
    gt_states = [to_frame(qn_frame(step)) for step in range(1, steps + 1)]
    mode_state_rows: dict[str, list[tuple[str, list[torch.Tensor]]]] = {mode: [("Ground truth", gt_states)] for mode in modes}
    metrics: list[dict[str, Any]] = []

    for spec in specs:
        print(f"[context-rollout] loading {spec.name}")
        model = load_model(spec, device)
        for mode in modes:
            current = qn_frame(0)
            states: list[torch.Tensor] = []
            for step in range(steps):
                if mode == "teacher_forced":
                    current = qn_frame(step)
                ctx = context_at(0) if mode == "static_context" else context_at(step)
                data = {**ctx, "qn_f": current}
                pred = predict_persistence(data) if spec.kind == "persistence" else predict(model, data)
                gt_next = qn_frame(step + 1)
                gt_prev = qn_frame(step)
                states.append(to_frame(pred))
                metrics.append(
                    {
                        "model": spec.name,
                        "mode": mode,
                        "step": step + 1,
                        "state_mae": mae(pred, gt_next),
                        "pred_rms": rms(pred),
                        "gt_rms": rms(gt_next),
                        "input_rms": rms(current),
                        "pred_tv": tv(pred),
                        "gt_tv": tv(gt_next),
                        "update_mae": mae(pred, current),
                        "true_update_mae": mae(gt_next, current),
                        "teacher_true_update_mae": mae(gt_next, gt_prev),
                        "update_cosine": cosine(pred - current, gt_next - current),
                    }
                )
                if mode != "teacher_forced":
                    current = pred
            mode_state_rows[mode].append((spec.name, states))
        if model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(args.output_dir / "rollout_context_metrics.csv", metrics)
    write_markdown(args.output_dir / "rollout_context_metrics.md", metrics)
    for mode, rows in mode_state_rows.items():
        plot_mode_grid(args.output_dir / f"{mode}_speed_rollout.png", rows, title=f"{mode.replace('_', ' ')} speed rollout")
    plot_lines(args.output_dir / "context_mode_state_mae.png", metrics, metric="state_mae", ylabel="State MAE")
    plot_lines(args.output_dir / "context_mode_pred_rms.png", metrics, metric="pred_rms", ylabel="Predicted RMS energy")
    plot_lines(args.output_dir / "context_mode_update_cosine.png", metrics, metric="update_cosine", ylabel="Update cosine")

    readme = [
        "# Rollout Context Diagnostics",
        "",
        "This compares three rollout modes:",
        "",
        "- `static_context`: keeps `ex_f/ex_g` frozen at the initial time while rolling out `qn_f`.",
        "- `dynamic_context`: advances `ex_f/ex_g` to the same time index as the rolled-out query, matching the dataset's one-step construction more closely.",
        "- `teacher_forced`: uses the ground-truth query frame at each step, so errors isolate one-step prediction quality without autoregressive drift.",
        "",
        "The image called `Input` in the main artifacts is the query input `qn_f`, not an in-context example. The context examples are `ex_f/ex_g` and are not shown in the main qualitative grid.",
        "",
        f"Split: `{args.split}`; sample index: `{sample_idx}`; start time: `{start_time}`; rollout steps: `{steps}`.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n")
    print(f"[context-rollout] wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
