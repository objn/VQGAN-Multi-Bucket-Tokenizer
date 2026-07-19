#!/usr/bin/env bash
# Downloads COCO train2017/val2017 and extracts them to LOCAL disk (default:
# $HOME/vqgan_data), then starts training via train_uv.sh (uv-managed) with
# --train-dir/--val-dir pointed at that local dataset. See train_with_coco_mini.sh
# for the plain-python equivalent.
#
# Why local and not the repo's data/ dir: on RunPod, the repo usually lives on a
# Network Volume (commonly mounted at /workspace) so it persists across pods --
# but writing 100k+ small .jpg files there is slow. $HOME is normally part of the
# pod's local Container Disk instead (fast, but wiped if the pod is terminated).
# Override with VQGAN_DATA_DIR if your setup differs, e.g. to keep the dataset on
# the network volume anyway (slower extraction, but survives pod termination):
#   VQGAN_DATA_DIR=data ./train_with_coco_mini_uv.sh
#
# Usage:
#   ./train_with_coco_mini_uv.sh
#   VQGAN_DATA_DIR=/some/local/path ./train_with_coco_mini_uv.sh
#   ./train_with_coco_mini_uv.sh --batch-size 2 --grad-accum-steps 8
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"

DATA_DIR="${VQGAN_DATA_DIR:-$HOME/vqgan_data}"
TRAIN_DIR="$DATA_DIR/train"
VAL_DIR="$DATA_DIR/val"

mkdir -p "$TRAIN_DIR" "$VAL_DIR"

fetch_and_extract() {
    local url="$1" target_dir="$2" zip_name="$3" zip_path="$DATA_DIR/$3"

    if [ -n "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        echo "==> $target_dir already has files, skipping download"
        return
    fi

    echo "==> Downloading $zip_name into $DATA_DIR..."
    wget -c "$url" -O "$zip_path"

    echo "==> Extracting $zip_name into $target_dir..."
    # -j (junk paths): COCO's zips nest everything under one top-level folder
    # (e.g. train2017/*.jpg) -- without -j this becomes $target_dir/train2017/*.jpg
    unzip -q -j "$zip_path" -d "$target_dir"

    rm -f "$zip_path"
}

fetch_and_extract "$TRAIN_URL" "$TRAIN_DIR" "train2017.zip"
fetch_and_extract "$VAL_URL" "$VAL_DIR" "val2017.zip"

echo "==> Dataset dir: $DATA_DIR"
echo "==> Starting training..."
./train_uv.sh --train-dir "$TRAIN_DIR" --val-dir "$VAL_DIR" "$@"
