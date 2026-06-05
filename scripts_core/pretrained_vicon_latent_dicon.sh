#!/bin/sh
# uv sync --extra cu128

# Train only the latent flow expert while keeping the pretrained VICON backbone frozen.
uv run python src/train.py --config-name=train_pdearena_vicon_dicon_latent_pretrained_frozen

echo "Done"
