#!/usr/bin/env bash
# Install the hf CLI and download ImageNet-1k into the shared workspace.
set -euo pipefail

LOCAL_DIR="${1:-/workspace/models/imagenet-1k}"

pip install -U "huggingface_hub[cli]" hf_transfer

if ! hf auth whoami >/dev/null 2>&1; then
    hf auth login
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

hf download ILSVRC/imagenet-1k \
    --type dataset \
    --local-dir "$LOCAL_DIR"
