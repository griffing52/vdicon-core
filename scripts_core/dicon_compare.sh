#!/bin/sh
uv sync --extra cu128
uv tree

# Train baseline ICON on KS.
uv run python src/train.py --config-name=train_dicon_baseline_icon trainer.max_steps=10 trainer.val_check_interval=5 trainer.limit_val_batches=5

# Train D-ICON (ICON + flow matching expert) on KS.
uv run python src/train.py --config-name=train_dicon trainer.max_steps=10 trainer.val_check_interval=5 trainer.limit_val_batches=5

# D-ICON validation logs include both:
# - <valid_dataset>/quest_qoi_v_icon
# - <valid_dataset>/quest_qoi_v_dicon
# - <valid_dataset>/quest_qoi_v_gain (positive means D-ICON improves over ICON)

echo "Done"
