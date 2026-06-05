#!/bin/sh
uv sync --extra cu128

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Overnight LTX-video baseline on PDEArena. Extra Hydra overrides can be passed after the script.
uv run python src/train.py --config-name=train_pdearena_ltx_vicon "$@"

echo "Done"
