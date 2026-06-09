#!/bin/sh
uv sync --extra cu128

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HF_HOME="${HF_HOME:-/home/griffing52/hf_cache}"

# Safer LTX-VICON retrain: qn_f spatial tokens + residual decoder.
uv run python src/train.py --config-name=train_pdearena_ltx_vicon_v2 "$@"

echo "Done"
