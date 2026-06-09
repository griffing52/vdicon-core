#!/bin/sh
uv sync --extra cu128

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

# Harder-horizon test: predict 10 PDEArena frames ahead instead of the near-identity next frame.
uv run python src/train.py --config-name=train_pdearena_vicon_transformer_video \
  data.target_time_offset=10 \
  tags='["pdearena","vicon_transformer_video","horizon10"]' \
  "$@"

echo "Done"
