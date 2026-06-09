#!/bin/sh
export HF_HOME="${HF_HOME:-/home/griffing52/hf_cache}"
uv run python scripts_core/ltx_vicon_v2_dry_run.py "$@"
