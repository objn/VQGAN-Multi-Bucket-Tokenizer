#!/usr/bin/env bash
# Runs VQGAN training via uv (manages the venv + deps from pyproject.toml/uv.lock).
# For environments without a pre-installed torch (e.g. local dev machines); see
# train.sh for the plain-python equivalent (no uv/.venv, for images that already ship
# torch/torchvision, like RunPod's PyTorch cluster templates).
# Assumes data/train and data/val already exist (see train_with_coco_mini_uv.sh if you
# need to fetch a dataset first).
#
# Usage:
#   ./train_uv.sh                              # uses configs/vqgan-multi.json defaults
#   ./train_uv.sh --resume-from runs/vqgan-multi/checkpoints/latest.pt
#   ./train_uv.sh --batch-size 2 --grad-accum-steps 8
#   VQGAN_NUM_GPUS=2 ./train_uv.sh             # DistributedDataParallel across 2 GPUs
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

NUM_GPUS="${VQGAN_NUM_GPUS:-1}"

if [ "$NUM_GPUS" -gt 1 ]; then
    uv run torchrun --standalone --nproc_per_node="$NUM_GPUS" -m vqgan.train "$@"
else
    uv run python -m vqgan.train "$@"
fi
