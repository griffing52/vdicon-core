from __future__ import annotations

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import hydra
import torch
from omegaconf import OmegaConf

from src.datasets.pdearena.pdearena_incomp_split import PDEArenaIncompSplitDataset


def move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def load_net_state(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    net_state = {
        key.removeprefix("net."): value
        for key, value in state_dict.items()
        if key.startswith("net.")
    }
    if not net_state:
        net_state = state_dict
    model.load_state_dict(net_state, strict=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run diagnostics for LTX-VICON v2.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional trained Lightning checkpoint to load.")
    parser.add_argument("--sample-idx", type=int, default=0, help="PDEArena sample index to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.create(
        {
            "paths": {"data_dir": str(repo_root / "data")},
            "model": OmegaConf.load(repo_root / "configs" / "model" / "ltx_vicon_v2.yaml"),
        }
    )
    model = hydra.utils.instantiate(cfg.model).to(device).eval()

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        load_net_state(model, checkpoint_path)
        print(f"loaded checkpoint: {checkpoint_path}")

    dataset = PDEArenaIncompSplitDataset(
        file_paths=str(repo_root / "data" / "pdearena_incomp" / "ns_incom_inhom_2d_512-*.h5"),
        ex_num=5,
        split="train",
    )
    sample = move_to_device(dataset[args.sample_idx], device)
    with torch.no_grad():
        pred = model(sample["data"], mode="test")["qn_pred"]
        diagnostics = model.diagnose(sample["data"], mode="test")

    label = sample["label"]
    copy_mae = (sample["data"]["qn_f"] - label).abs().mean()
    pred_mae = (pred - label).abs().mean()

    print(f"device: {device}")
    print(f"sample index: {args.sample_idx}")
    print(f"prediction shape: {tuple(pred.shape)}")
    print(f"prediction finite: {bool(torch.isfinite(pred).all())}")
    print(f"prediction MAE: {float(pred_mae.detach().cpu()):.6f}")
    print(f"copy-qn_f MAE: {float(copy_mae.detach().cpu()):.6f}")
    for key, value in diagnostics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
