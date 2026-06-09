#!/bin/sh
uv sync --extra cu128

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HF_HOME="${HF_HOME:-/home/griffing52/hf_cache}"

# Harder-horizon test for LTX-VICON v2. This should expose whether the model is only learning persistence.
uv run python src/train.py --config-name=train_pdearena_ltx_vicon_v2 \
  data.target_time_offset=10 \
  tags='["pdearena","ltx_video_v2","horizon10"]' \
  "$@"

echo "Done"
