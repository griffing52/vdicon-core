from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Any

import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts_core.generate_paper_artifacts import (  # noqa: E402
    auto_discover_models,
    curated_baselines,
    dedupe_specs,
    load_checkpoint_model,
    load_lightning_model,
    load_pretrained_vicon,
    move_to_device,
    predict,
    predict_persistence,
    write_csv,
)
from src.datasets.pdearena.pdearena_incomp_split import PDEArenaIncompSplitDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose persistence/identity shortcuts on PDEArena next-frame prediction.")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "paper_artifacts" / "identity_diagnostics")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50])
    parser.add_argument("--time-bin-size", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-auto-models", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    return parser.parse_args()


def make_dataset(split: str, target_time_offset: int = 1) -> PDEArenaIncompSplitDataset:
    return PDEArenaIncompSplitDataset(
        file_paths=str(repo_root / "data" / "pdearena_incomp" / "ns_incom_inhom_2d_512-*.h5"),
        ex_num=5,
        split=split,
        target_time_offset=target_time_offset,
    )


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu().flatten()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred.detach().float().cpu() - target.detach().float().cpu()).abs().mean())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff = pred.detach().float().cpu() - target.detach().float().cpu()
    return float(torch.sqrt(diff.square().mean()))


def tv(x: torch.Tensor) -> float:
    x = x.detach().float().cpu()
    while x.ndim > 4:
        x = x[0]
    dx = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dy = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return float(dx + dy)


def cosine_update(pred: torch.Tensor, inp: torch.Tensor, target: torch.Tensor) -> float:
    pred_update = flatten(pred - inp)
    true_update = flatten(target - inp)
    denom = pred_update.norm() * true_update.norm() + 1e-12
    return float(torch.dot(pred_update, true_update) / denom)


def summarize(rows: list[dict[str, Any]], group_key: str, numeric_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    out = []
    for name, group in groups.items():
        summary: dict[str, Any] = {group_key: name, "n": len(group)}
        for key in numeric_keys:
            vals = torch.tensor([float(row[key]) for row in group], dtype=torch.float32)
            summary[f"{key}_mean"] = float(vals.mean())
            summary[f"{key}_std"] = float(vals.std(unbiased=False))
        out.append(summary)
    return out


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("No rows.\n")
        return
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row[col]
            vals.append(f"{val:.6f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def horizon_persistence(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for horizon in args.horizons:
        dataset = make_dataset(args.split, target_time_offset=horizon)
        n = min(args.num_samples, len(dataset))
        for idx in range(n):
            sample = dataset[idx]
            inp = sample["data"]["qn_f"]
            target = sample["label"]
            qn_traj, time_idx = dataset._samples[idx]
            rows.append(
                {
                    "horizon": horizon,
                    "sample_idx": idx,
                    "traj_idx": qn_traj,
                    "time_idx": time_idx,
                    "mae": mae(inp, target),
                    "rmse": rmse(inp, target),
                    "target_tv": tv(target),
                    "input_tv": tv(inp),
                }
            )
    return rows


def time_bin_persistence(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = make_dataset(args.split, target_time_offset=1)
    rows = []
    n = min(len(dataset), max(args.num_samples, 256))
    for idx in range(n):
        sample = dataset[idx]
        _, time_idx = dataset._samples[idx]
        rows.append(
            {
                "time_bin": int(time_idx // args.time_bin_size),
                "time_start": int((time_idx // args.time_bin_size) * args.time_bin_size),
                "sample_idx": idx,
                "mae": mae(sample["data"]["qn_f"], sample["label"]),
                "target_tv": tv(sample["label"]),
                "input_tv": tv(sample["data"]["qn_f"]),
            }
        )
    return rows


def load_model_for_spec(spec, device: torch.device):
    if spec.kind == "persistence":
        return None
    if spec.kind == "pretrained_vicon":
        return load_pretrained_vicon(device)
    if spec.kind == "lightning":
        return load_lightning_model(spec, device)
    return load_checkpoint_model(spec, device)


def model_identity_stats(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.skip_models:
        return []
    device = torch.device(args.device)
    dataset = make_dataset(args.split, target_time_offset=1)
    specs = curated_baselines()
    if args.include_auto_models:
        specs.extend(auto_discover_models())
    specs = dedupe_specs(specs)
    rows = []
    n = min(args.num_samples, len(dataset))
    for spec in specs:
        print(f"[identity] loading {spec.name}")
        model = load_model_for_spec(spec, device)
        for idx in range(n):
            sample = move_to_device(dataset[idx], device)
            inp = sample["data"]["qn_f"]
            target = sample["label"]
            pred = predict_persistence(sample["data"]) if spec.kind == "persistence" else predict(model, sample["data"])
            true_update = mae(target, inp)
            pred_update = mae(pred, inp)
            rows.append(
                {
                    "model": spec.name,
                    "sample_idx": idx,
                    "pred_mae": mae(pred, target),
                    "persistence_mae": true_update,
                    "pred_update_mae": pred_update,
                    "update_ratio": pred_update / (true_update + 1e-12),
                    "update_cosine": cosine_update(pred, inp, target),
                    "pred_tv": tv(pred),
                    "input_tv": tv(inp),
                    "target_tv": tv(target),
                    "pred_tv_ratio": tv(pred) / (tv(target) + 1e-12),
                }
            )
        if model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    horizon_rows = horizon_persistence(args)
    write_csv(args.output_dir / "persistence_by_horizon_samples.csv", horizon_rows)
    horizon_summary = summarize(horizon_rows, "horizon", ["mae", "rmse", "target_tv", "input_tv"])
    write_csv(args.output_dir / "persistence_by_horizon.csv", horizon_summary)
    write_markdown(args.output_dir / "persistence_by_horizon.md", horizon_summary)

    time_rows = time_bin_persistence(args)
    write_csv(args.output_dir / "persistence_by_time_bin_samples.csv", time_rows)
    time_summary = summarize(time_rows, "time_bin", ["mae", "target_tv", "input_tv"])
    for row in time_summary:
        row["time_start"] = int(row["time_bin"]) * args.time_bin_size
    write_csv(args.output_dir / "persistence_by_time_bin.csv", time_summary)
    write_markdown(args.output_dir / "persistence_by_time_bin.md", time_summary)

    model_rows = model_identity_stats(args)
    write_csv(args.output_dir / "model_identity_samples.csv", model_rows)
    model_summary = summarize(
        model_rows,
        "model",
        ["pred_mae", "persistence_mae", "pred_update_mae", "update_ratio", "update_cosine", "pred_tv", "input_tv", "target_tv", "pred_tv_ratio"],
    )
    write_csv(args.output_dir / "model_identity_summary.csv", model_summary)
    write_markdown(args.output_dir / "model_identity_summary.md", model_summary)
    print(f"[identity] wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
