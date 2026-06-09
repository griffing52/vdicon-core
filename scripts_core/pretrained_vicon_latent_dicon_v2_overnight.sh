#!/bin/sh
# Train a video-generation-style residual flow-matching head conditioned on
# frozen pretrained VICON hidden states and sinusoidal flow-time embeddings.
uv run python src/train.py --config-name=train_pdearena_vicon_dicon_latent_pretrained_v2_overnight "$@"

echo "Done"
