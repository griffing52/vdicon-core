#!/bin/sh
uv sync --extra cu128
uv tree

# KS D-ICON flow-matching run.
uv run python src/train.py --config-name=train_dicon trainer.max_steps=10 trainer.val_check_interval=5 trainer.limit_val_batches=5

# WENO transfer run with a slightly larger transformer.
uv run python src/train.py --config-name=train_dicon_weno trainer.max_steps=10 trainer.val_check_interval=5 trainer.limit_val_batches=5

# Long-prompt ablation on KS.
uv run python src/train.py --config-name=train_dicon_long_prompt trainer.max_steps=10 trainer.val_check_interval=5 trainer.limit_val_batches=5

# Fast CPU smoke test.
export TORCH_COMPILE_DISABLE=1
uv run python src/train.py --config-name=train_dicon_smoke

echo "Done"
