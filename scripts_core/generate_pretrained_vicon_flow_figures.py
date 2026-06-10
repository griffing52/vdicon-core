from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROJECT_ROOT", str(repo_root))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts_core.generate_paper_artifacts import (  # noqa: E402
    ModelSpec,
    field_maps,
    load_lightning_model,
    move_to_device,
    predict,
    scalar_metrics,
    to_frame,
)
from src.datasets.pdearena.pdearena_incomp_split import PDEArenaIncompSplitDataset  # noqa: E402

FLOW_CKPT = repo_root / "logs/train/runs/2026-06-09_07-21-52-572951/checkpoints/step_100000.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pretrained VICON flow-v2 architecture and denoising figures.")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "paper_artifacts")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=32)
    parser.add_argument("--snapshots", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=FLOW_CKPT)
    return parser.parse_args()


def squeeze_frame(tensor: torch.Tensor) -> torch.Tensor:
    frame = tensor.detach().float().cpu()
    while frame.ndim > 3:
        frame = frame[0]
    return frame


def speed_map(tensor: torch.Tensor) -> torch.Tensor:
    return field_maps(squeeze_frame(tensor))[0]


def flow_conditioning(model: torch.nn.Module, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    net = getattr(model, "net", model)
    pred_icon = net.predict_backbone(data=data, mode="test", need_weights=False)
    cond = net.extract_hidden(data=data, mode="test")
    baseline = pred_icon["qn_pred"]
    target_size = tuple(int(x) for x in data["qn_f"].shape[-2:])
    return cond, baseline, target_size


def flow_trajectory(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    num_steps: int,
    snapshot_count: int,
) -> tuple[list[tuple[int, float, torch.Tensor]], torch.Tensor, torch.Tensor]:
    net = getattr(model, "net", model)
    cond, baseline, target_size = flow_conditioning(model, data)
    cond = net._squeeze_sample_dim(cond)
    baseline = net._squeeze_sample_dim(baseline)

    batch_size = cond.shape[0]
    dtype = cond.dtype
    device = cond.device
    dt = 1.0 / num_steps
    x = torch.randn(batch_size, baseline.shape[1], target_size[0], target_size[1], device=device, dtype=dtype)

    wanted = sorted(set(round(i * num_steps / (snapshot_count - 1)) for i in range(snapshot_count)))
    snapshots: list[tuple[int, float, torch.Tensor]] = []
    if 0 in wanted:
        snapshots.append((0, 0.0, x.detach().clone() + baseline))

    with torch.no_grad():
        for step in range(num_steps):
            t_value = step / num_steps
            t = torch.full((batch_size,), t_value, device=device, dtype=dtype)
            v1 = net.flow_expert(x_t=x, cond=cond, t=t)
            x_mid = x + dt * v1
            t_next_value = min(t_value + dt, 1.0)
            t_next = torch.full((batch_size,), t_next_value, device=device, dtype=dtype)
            v2 = net.flow_expert(x_t=x_mid, cond=cond, t=t_next)
            x = x + 0.5 * dt * (v1 + v2)
            if step + 1 in wanted:
                snapshots.append((step + 1, t_next_value, x.detach().clone() + baseline))

    pred = snapshots[-1][2]
    return snapshots, pred, baseline


def plot_flow_trajectory(path: Path, snapshots: list[tuple[int, float, torch.Tensor]], target: torch.Tensor) -> None:
    maps = [speed_map(frame) for _, _, frame in snapshots]
    target_speed = speed_map(target)
    all_maps = maps + [target_speed]
    vmin = float(torch.stack(all_maps).quantile(0.01))
    vmax = float(torch.stack(all_maps).quantile(0.99))
    err_maps = [(m - target_speed).abs() for m in maps]
    err_vmax = float(torch.stack(err_maps).quantile(0.99))

    cols = len(snapshots)
    fig, axes = plt.subplots(2, cols, figsize=(2.2 * cols, 4.6), constrained_layout=True)
    for col, ((step, t_value, _), map_tensor, err_tensor) in enumerate(zip(snapshots, maps, err_maps)):
        ax = axes[0, col]
        im = ax.imshow(map_tensor, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"step {step}\nt={t_value:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("speed", fontsize=10)
        ax = axes[1, col]
        ax.imshow(err_tensor, cmap="magma", vmin=0.0, vmax=err_vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("abs. error", fontsize=10)
    fig.colorbar(im, ax=axes[0, :], fraction=0.025, pad=0.01, label="speed")
    fig.suptitle("Pretrained VICON flow-v2 Heun trajectory from noise to prediction", fontsize=13, weight="bold")
    fig.savefig(path, dpi=230)
    plt.close(fig)


def add_box(ax, xy, width, height, text, fc, ec="#27313f", fontsize=10, weight="normal"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, transform=ax.transAxes)


def add_arrow(ax, start, end, color="#27313f", dashed=False, lw=1.8):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        color=color,
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)


def plot_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
    ax.set_axis_off()
    ax.text(0.5, 0.96, "Pretrained VICON flow-v2 system", ha="center", va="center", fontsize=15, weight="bold", transform=ax.transAxes)

    add_box(ax, (0.04, 0.63), 0.17, 0.16, "PDEArena prompt\n5 example transitions\n+ query field", "#eef5f1")
    add_box(ax, (0.27, 0.70), 0.18, 0.12, "channel lift\n2 velocity -> 7 channels", "#f7f7f7")
    add_box(ax, (0.50, 0.70), 0.19, 0.12, "frozen pretrained\nVICON transformer", "#eaf0ff", weight="bold")
    add_box(ax, (0.75, 0.70), 0.18, 0.12, "pretrained decoder\nbaseline $\\hat{x}_0$", "#eaf0ff")

    add_box(ax, (0.50, 0.43), 0.19, 0.12, "hidden states\nconditioning map $c$", "#efe9ff")
    add_box(ax, (0.10, 0.20), 0.15, 0.12, "Gaussian noise\n$residual z_0$", "#f1f1f1")
    add_box(ax, (0.34, 0.20), 0.20, 0.12, "time embedding\n$t \\in [0,1]$", "#fff4df")
    add_box(ax, (0.62, 0.20), 0.20, 0.12, "residual flow head\n$v_\\theta(z_t,c,t)$", "#ffe8df", weight="bold")
    add_box(ax, (0.79, 0.43), 0.17, 0.12, "Heun ODE sampling\n32 steps", "#ffe8df")
    add_box(ax, (0.79, 0.20), 0.17, 0.12, "final prediction\n$\\hat{x}=\\hat{x}_0+r$", "#e9f7ef", weight="bold")

    add_arrow(ax, (0.21, 0.71), (0.27, 0.76))
    add_arrow(ax, (0.45, 0.76), (0.50, 0.76))
    add_arrow(ax, (0.69, 0.76), (0.75, 0.76))
    add_arrow(ax, (0.59, 0.70), (0.59, 0.55), color="#6d4bb3", dashed=True)
    add_arrow(ax, (0.25, 0.26), (0.62, 0.26), color="#b55b3c")
    add_arrow(ax, (0.54, 0.26), (0.62, 0.26), color="#b55b3c")
    add_arrow(ax, (0.69, 0.43), (0.68, 0.32), color="#6d4bb3", dashed=True)
    add_arrow(ax, (0.82, 0.70), (0.86, 0.55), color="#3a63ad")
    add_arrow(ax, (0.82, 0.32), (0.86, 0.43), color="#b55b3c")
    add_arrow(ax, (0.875, 0.43), (0.875, 0.32), color="#b55b3c")

    ax.text(0.47, 0.08, "Training objective: cosine-bridge conditional flow matching on residual target $r = x_{t+1} - \\hat{x}_0$; the released VICON backbone stays frozen.", ha="center", fontsize=10, color="#333333", transform=ax.transAxes)
    fig.savefig(path, dpi=230)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(7)

    dataset = PDEArenaIncompSplitDataset(
        file_paths=str(repo_root / "data" / "pdearena_incomp" / "*.h5"),
        ex_num=5,
        split=args.split,
        target_time_offset=1,
    )
    sample_idx = min(args.sample_idx, len(dataset) - 1)
    sample = move_to_device(dataset[sample_idx], device)
    spec = ModelSpec("Pretrained VICON flow v2", args.checkpoint, "lightning")
    model = load_lightning_model(spec, device)

    snapshots, pred, baseline = flow_trajectory(model, sample["data"], args.flow_steps, args.snapshots)
    target = sample["label"]
    metrics = scalar_metrics(pred, target)
    baseline_metrics = scalar_metrics(baseline[:, None] if baseline.ndim == 4 else baseline, target)

    plot_flow_trajectory(args.output_dir / "pretrained_vicon_flow_v2_denoising_steps.png", snapshots, target)
    plot_architecture(args.output_dir / "pretrained_vicon_flow_v2_architecture.png")

    summary = args.output_dir / "pretrained_vicon_flow_v2_figures.md"
    summary.write_text(
        "# Pretrained VICON flow-v2 figures\n\n"
        f"Checkpoint: `{args.checkpoint}`\n\n"
        f"Split: `{args.split}`; sample index: `{sample_idx}`; flow steps: `{args.flow_steps}`; snapshots: `{len(snapshots)}`.\n\n"
        f"Final sampled MAE: `{metrics['mae']:.5f}`; baseline MAE before residual flow: `{baseline_metrics['mae']:.5f}`.\n"
    )
    print(f"[flow-v2-figures] wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
