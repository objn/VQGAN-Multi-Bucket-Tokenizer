#!/usr/bin/env bash
# Runs VQGAN training with plain python -- no uv/.venv. For environments that already
# have torch/torchvision installed at the system level (e.g. RunPod's PyTorch cluster
# templates); see train_uv.sh for the uv-managed equivalent.
# Assumes data/train and data/val already exist (see train_with_coco_mini.sh if you
# need to fetch a dataset first) and the remaining deps are installed, e.g.:
#   pip install lpips scipy tensorboard tqdm pillow
#
# Usage:
#   ./train.sh                              # uses configs/vqgan-multi.json defaults
#   ./train.sh --resume-from runs/vqgan-multi/checkpoints/latest.pt
#   ./train.sh --batch-size 2 --grad-accum-steps 8
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python -m vqgan.train "$@"
