#!/usr/bin/env bash
# Runs VQGAN training. Assumes data/train and data/val already exist
# (see train_with_coco_mini.sh if you need to fetch a dataset first).
#
# Usage:
#   ./train.sh                              # uses configs/vqgan-multi.json defaults
#   ./train.sh --resume-from runs/vqgan-multi/checkpoints/latest.pt
#   ./train.sh --batch-size 2 --grad-accum-steps 8
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

uv run python -m vqgan.train "$@"
