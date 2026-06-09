from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper artifacts for the PDEArena persistence/identity-shortcut finding.")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "paper_artifacts" / "identity_paper")
    parser.add_argument("--diagnostics-dir", type=Path, default=repo_root / "paper_artifacts" / "identity_paper" / "diagnostics")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--model-num-samples", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-diagnostics", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("No rows.\n")
        return
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("% No rows.\n")
        return
    columns = list(rows[0].keys())
    newline = r"\\"
    lines = ["\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}", "\\toprule"]
    lines.append(" & ".join(columns).replace("_", "\\_") + f" {newline}")
    lines.append("\\midrule")
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(values) + f" {newline}")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def f(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "nan")
    return float(value) if value not in (None, "") else math.nan


def run_diagnostics(args: argparse.Namespace) -> None:
    if args.skip_diagnostics:
        return
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "scripts_core" / "pdearena_identity_diagnostics.py"),
        "--output-dir",
        str(args.diagnostics_dir),
        "--split",
        args.split,
        "--num-samples",
        str(args.num_samples),
        "--horizons",
        *[str(horizon) for horizon in args.horizons],
        "--skip-models",
    ]
    print("[identity-paper] running data diagnostics")
    subprocess.run(cmd, check=True, cwd=repo_root)

    model_dir = args.output_dir / "model_diagnostics"
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "scripts_core" / "pdearena_identity_diagnostics.py"),
        "--output-dir",
        str(model_dir),
        "--split",
        args.split,
        "--num-samples",
        str(args.model_num_samples),
        "--horizons",
        *[str(horizon) for horizon in args.horizons],
        "--include-auto-models",
        "--device",
        args.device,
    ]
    print("[identity-paper] running model diagnostics")
    subprocess.run(cmd, check=True, cwd=repo_root)


def plot_horizon(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: f(row, "horizon"))
    x = [f(row, "horizon") for row in rows]
    y = [f(row, "mae_mean") for row in rows]
    yerr = [f(row, "mae_std") for row in rows]
    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2.0, capsize=3, color="#4c78a8")
    ax.set_xscale("log")
    ax.set_xlabel("Forecast horizon (frames)")
    ax.set_ylabel("Persistence MAE")
    ax.set_title("Persistence becomes a weaker baseline at longer horizons")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_time_bins(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: f(row, "time_start"))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    x = [f(row, "time_start") for row in rows]
    ax1.plot(x, [f(row, "mae_mean") for row in rows], marker="o", color="#e45756", label="Persistence MAE")
    ax1.set_xlabel("Trajectory time index")
    ax1.set_ylabel("Persistence MAE", color="#e45756")
    ax1.tick_params(axis="y", labelcolor="#e45756")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, [f(row, "target_tv_mean") for row in rows], marker="s", color="#54a24b", label="Target TV")
    ax2.set_ylabel("Target total variation", color="#54a24b")
    ax2.tick_params(axis="y", labelcolor="#54a24b")
    ax1.set_title("One-step changes collapse after early trajectory times")
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_model_bars(path: Path, rows: list[dict[str, Any]], metric: str, ylabel: str, title: str, color: str) -> None:
    rows = sorted(rows, key=lambda row: f(row, metric))
    names = [row["model"] for row in rows]
    values = [f(row, metric) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8.0, 1.0 * len(names)), 4.8), constrained_layout=True)
    ax.bar(range(len(names)), values, color=color)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_pred_vs_persistence(path: Path, rows: list[dict[str, Any]]) -> None:
    names = [row["model"] for row in rows]
    pred = [f(row, "pred_mae_mean") for row in rows]
    pers = [f(row, "persistence_mae_mean") for row in rows]
    width = 0.38
    x = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.05 * len(names)), 4.8), constrained_layout=True)
    ax.bar([i - width / 2 for i in x], pers, width=width, label="Persistence", color="#bab0ab")
    ax.bar([i + width / 2 for i in x], pred, width=width, label="Model", color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Model error versus persistence on the same samples")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_update_scatter(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    for row in rows:
        ax.scatter(f(row, "update_ratio_mean"), f(row, "update_cosine_mean"), s=70)
        ax.annotate(row["model"], (f(row, "update_ratio_mean"), f(row, "update_cosine_mean")), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axvline(1.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Predicted update magnitude / true update magnitude")
    ax.set_ylabel("Cosine alignment with true update")
    ax.set_title("Identity shortcut diagnosis: magnitude versus direction")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def compact_horizon_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "horizon": int(f(row, "horizon")),
            "persistence_mae": f(row, "mae_mean"),
            "persistence_rmse": f(row, "rmse_mean"),
            "target_tv": f(row, "target_tv_mean"),
        }
        for row in sorted(rows, key=lambda row: f(row, "horizon"))
    ]


def compact_model_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = []
    for row in rows:
        keep.append(
            {
                "model": row["model"],
                "model_mae": f(row, "pred_mae_mean"),
                "persistence_mae": f(row, "persistence_mae_mean"),
                "update_ratio": f(row, "update_ratio_mean"),
                "update_cosine": f(row, "update_cosine_mean"),
                "tv_ratio": f(row, "pred_tv_ratio_mean"),
            }
        )
    return keep


def write_interpretation(path: Path, horizon_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]]) -> None:
    h1 = next(row for row in horizon_rows if int(f(row, "horizon")) == 1)
    h10 = next(row for row in horizon_rows if int(f(row, "horizon")) == 10)
    h20 = next(row for row in horizon_rows if int(f(row, "horizon")) == 20)
    ltx = next((row for row in model_rows if row["model"] == "LTX-VICON"), None)
    vt = next((row for row in model_rows if row["model"] == "VICON transformer video"), None)
    pre = next((row for row in model_rows if row["model"] == "Pretrained VICON"), None)
    lines = [
        "# Identity/Persistence Paper Artifacts",
        "",
        "This artifact set reframes the one-step PDEArena results around a persistence baseline.",
        "",
        "## Main Takeaway",
        "",
        f"At horizon 1, raw persistence has MAE `{f(h1, 'mae_mean'):.6f}`. At horizon 10 it rises to `{f(h10, 'mae_mean'):.6f}`, and at horizon 20 it rises to `{f(h20, 'mae_mean'):.6f}`. This means the default one-step task is highly persistence-friendly, while longer horizons provide a cleaner test of learned dynamics.",
        "",
        "## Model Behavior",
        "",
    ]
    if ltx is not None:
        lines.append(f"- LTX-VICON has update ratio `{f(ltx, 'update_ratio_mean'):.3f}`, so it changes the input much less than the true update on the diagnostic subset.")
    if pre is not None:
        lines.append(f"- Pretrained VICON has update cosine `{f(pre, 'update_cosine_mean'):.3f}`, indicating better directionality of the correction.")
    if vt is not None:
        lines.append(f"- VICON transformer video has update cosine `{f(vt, 'update_cosine_mean'):.3f}`, the strongest update alignment in this diagnostic run.")
    lines.extend(
        [
            "",
            "## Recommended Paper Framing",
            "",
            "Report persistence as a first-class baseline. Treat horizon-1 results as a near-persistence forecasting regime, not as definitive evidence of dynamics learning. Use horizon-10 or horizon-20 training/evaluation to test whether the architectures learn nontrivial temporal evolution.",
            "",
            "## Generated Files",
            "",
            "- `persistence_by_horizon.png`: how the persistence baseline weakens as forecast horizon grows.",
            "- `persistence_by_time_bin.png`: how one-step changes collapse over trajectory time.",
            "- `model_vs_persistence_mae.png`: model MAE versus persistence MAE on identical samples.",
            "- `model_update_ratio.png`: how much each model moves relative to the true update.",
            "- `model_update_cosine.png`: alignment between model update and true update.",
            "- `model_update_scatter.png`: update magnitude versus direction in one panel.",
            "- `horizon_summary.{csv,md,tex}` and `model_identity_summary.{csv,md,tex}`: compact paper tables.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir = args.diagnostics_dir.resolve()
    run_diagnostics(args)

    data_diag = args.diagnostics_dir
    model_diag = args.output_dir / "model_diagnostics"
    horizon_rows = read_csv(data_diag / "persistence_by_horizon.csv")
    time_rows = read_csv(data_diag / "persistence_by_time_bin.csv")
    model_rows = read_csv(model_diag / "model_identity_summary.csv")

    shutil.copy2(data_diag / "persistence_by_horizon.csv", args.output_dir / "raw_persistence_by_horizon.csv")
    shutil.copy2(data_diag / "persistence_by_time_bin.csv", args.output_dir / "raw_persistence_by_time_bin.csv")
    shutil.copy2(model_diag / "model_identity_summary.csv", args.output_dir / "raw_model_identity_summary.csv")

    horizon_table = compact_horizon_table(horizon_rows)
    model_table = compact_model_table(model_rows)
    write_csv(args.output_dir / "horizon_summary.csv", horizon_table)
    write_markdown_table(args.output_dir / "horizon_summary.md", horizon_table)
    write_latex_table(args.output_dir / "horizon_summary.tex", horizon_table)
    write_csv(args.output_dir / "model_identity_summary.csv", model_table)
    write_markdown_table(args.output_dir / "model_identity_summary.md", model_table)
    write_latex_table(args.output_dir / "model_identity_summary.tex", model_table)

    plot_horizon(args.output_dir / "persistence_by_horizon.png", horizon_rows)
    plot_time_bins(args.output_dir / "persistence_by_time_bin.png", time_rows)
    plot_pred_vs_persistence(args.output_dir / "model_vs_persistence_mae.png", model_rows)
    plot_model_bars(args.output_dir / "model_update_ratio.png", model_rows, "update_ratio_mean", "Update ratio", "Predicted update magnitude relative to true update", "#f58518")
    plot_model_bars(args.output_dir / "model_update_cosine.png", model_rows, "update_cosine_mean", "Update cosine", "Directional alignment with the true update", "#54a24b")
    plot_model_bars(args.output_dir / "model_tv_ratio.png", model_rows, "pred_tv_ratio_mean", "Predicted TV / target TV", "Output sharpness/smoothness proxy", "#b279a2")
    plot_update_scatter(args.output_dir / "model_update_scatter.png", model_rows)
    write_interpretation(args.output_dir / "interpretation.md", horizon_rows, model_rows)
    print(f"[identity-paper] wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
