"""Stage 1: scan raw images, filter/resize/pad, cache to a memmap array + metadata.

Usage:
    python scripts/preprocess.py --root images --out data/processed
"""

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqgan.config import PreprocessConfig
from vqgan.data.preprocessing import (
    discover_images,
    is_valid_image,
    passes_resolution_filter,
    resize_and_crop,
)


def parse_args(argv=None) -> PreprocessConfig:
    defaults = PreprocessConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=defaults.root)
    parser.add_argument("--out", dest="out_dir", default=defaults.out_dir)
    parser.add_argument("--canvas-size", type=int, default=defaults.canvas_size)
    parser.add_argument("--min-short-side", type=int, default=defaults.min_short_side)
    parser.add_argument("--downsample-factor", type=int, default=defaults.downsample_factor)
    parser.add_argument("--val-frac", type=float, default=defaults.val_frac)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args(argv)
    return PreprocessConfig(**vars(args))


def main(argv=None):
    cfg = parse_args(argv)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_images(cfg.root)
    print(f"discovered {len(discovered)} candidate images under {cfg.root}")

    kept_records = []
    canvases = []
    n_corrupt = 0
    n_too_small = 0

    for item in tqdm(discovered, desc="preprocessing"):
        if not is_valid_image(item.path):
            n_corrupt += 1
            continue
        try:
            with Image.open(item.path) as im:
                if not passes_resolution_filter(im, cfg.min_short_side):
                    n_too_small += 1
                    continue
                orig_w, orig_h = im.size
                canvas, final_h, final_w = resize_and_crop(
                    im, canvas_size=cfg.canvas_size, downsample_factor=cfg.downsample_factor
                )
        except (OSError, ValueError):
            n_corrupt += 1
            continue

        canvases.append(canvas)
        kept_records.append(
            {
                "index": len(kept_records),
                "filename": item.path.name,
                "account": item.account,
                "orig_w": orig_w,
                "orig_h": orig_h,
                "final_w": final_w,
                "final_h": final_h,
                "grid_w": final_w // cfg.downsample_factor,
                "grid_h": final_h // cfg.downsample_factor,
                "aspect_ratio": orig_w / orig_h,
            }
        )

    n_kept = len(kept_records)
    print(
        f"kept {n_kept} images "
        f"(dropped {n_corrupt} corrupt/unreadable, {n_too_small} below min-short-side)"
    )
    if n_kept == 0:
        print("nothing to write, exiting")
        return

    pixels_path = out_dir / "pixels.npy"
    pixels = np.lib.format.open_memmap(
        pixels_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n_kept, cfg.canvas_size, cfg.canvas_size, 3),
    )
    for i, canvas in enumerate(canvases):
        pixels[i] = canvas
    pixels.flush()
    del pixels

    metadata_path = out_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for record in kept_records:
            f.write(json.dumps(record) + "\n")

    # Prep everything first, then take a plain random val_frac slice of all
    # images (no account grouping) — simple global shuffle-and-split.
    rng = random.Random(cfg.seed)
    idx = list(range(n_kept))
    rng.shuffle(idx)
    n_val = round(n_kept * cfg.val_frac)
    val_idx = sorted(idx[:n_val])
    train_idx = sorted(idx[n_val:])

    splits_path = out_dir / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump({"train": train_idx, "val": val_idx}, f)

    print(f"train: {len(train_idx)}  val: {len(val_idx)}")
    print(f"wrote {pixels_path}, {metadata_path}, {splits_path}")

    config_path = out_dir / "preprocess_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


if __name__ == "__main__":
    main()
