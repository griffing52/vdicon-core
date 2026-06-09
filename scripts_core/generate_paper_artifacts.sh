#!/bin/sh
export HF_HOME="${HF_HOME:-/home/griffing52/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python scripts_core/generate_paper_artifacts.py "$@"
