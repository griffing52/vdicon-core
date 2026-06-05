#!/bin/sh
uv sync --extra cu128

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train VICON-style video adapters/decoder while keeping the pretrained VICON transformer frozen.
uv run python src/train.py --config-name=train_pdearena_vicon_transformer_video "$@"

echo "Done"
