#!/bin/sh
# Train the flow expert on a 10-frame-ahead target while keeping the pretrained VICON backbone frozen.
uv run python src/train.py --config-name=train_pdearena_vicon_dicon_latent_pretrained_frozen \
  data.target_time_offset=10 \
  tags='["pdearena","pretrained_vicon_latent_dicon","horizon10"]' \
  "$@"

echo "Done"
