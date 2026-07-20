#!/usr/bin/env bash
# Runs VQGAN training with plain python -- no uv/.venv. For environments that already
# have torch/torchvision installed at the system level (e.g. RunPod's PyTorch cluster
# templates); see train_uv.sh for the uv-managed equivalent.
# Assumes data/train and data/val already exist (see train_with_coco_mini.sh if you
# need to fetch a dataset first) and deps are installed, e.g.:
#   pip install -r requirements.txt
#
# Usage:
#   ./train.sh                              # uses configs/vqgan-multi.json defaults
#   ./train.sh --resume-from runs/vqgan-multi/checkpoints/latest.pt
#   ./train.sh --target-epochs 10           # per-bucket batch_size/accumulation_steps live in the config JSON
#   VQGAN_NUM_GPUS=2 ./train.sh             # DistributedDataParallel across 2 GPUs
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

NUM_GPUS="${VQGAN_NUM_GPUS:-1}"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" -m vqgan.train "$@"
else
    python -m vqgan.train "$@"
fi
