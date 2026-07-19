#!/usr/bin/env bash
# Downloads COCO train2017/val2017 into data/train and data/val (skipped if already
# present), then starts training via train_uv.sh (uv-managed). See
# train_with_coco_mini.sh for the plain-python equivalent. Any extra args are
# forwarded to train_uv.sh.
#
# Usage:
#   ./train_with_coco_mini_uv.sh
#   ./train_with_coco_mini_uv.sh --batch-size 2 --grad-accum-steps 8
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"

DATA_DIR="data"
TRAIN_DIR="$DATA_DIR/train"
VAL_DIR="$DATA_DIR/val"

mkdir -p "$TRAIN_DIR" "$VAL_DIR"

fetch_and_extract() {
    local url="$1" target_dir="$2" zip_name="$3"

    if [ -n "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        echo "==> $target_dir already has files, skipping download"
        return
    fi

    echo "==> Downloading $zip_name..."
    wget -c "$url" -O "$DATA_DIR/$zip_name"

    echo "==> Extracting $zip_name into $target_dir..."
    unzip -q "$DATA_DIR/$zip_name" -d "$target_dir"

    rm -f "$DATA_DIR/$zip_name"
}

fetch_and_extract "$TRAIN_URL" "$TRAIN_DIR" "train2017.zip"
fetch_and_extract "$VAL_URL" "$VAL_DIR" "val2017.zip"

echo "==> Starting training..."
./train_uv.sh "$@"
