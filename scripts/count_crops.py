"""How many training crops the current index actually contains.

Doesn't decode any pixels: only enough of each image's header to read its
(width, height) — the same trick build_index.py's `verified_size` uses for
the folder source, applied here to the parquet source too, where dimensions
aren't already known. The tile count itself is pure arithmetic
(`vqgan.data.tiling.count_tiles`), matching what CropDataset would produce
without touching the RNG that drives jitter — jitter only moves where the
tiles land each pass, never how many of them there are.

Usage:
    python scripts/count_crops.py
    python scripts/count_crops.py --source folder --tile-size 256
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqgan.config import DataConfig, VQGANTrainConfig
from vqgan.data.sources import PARQUET_SPLITS, build_shards, iter_sizes, shard_total
from vqgan.data.tiling import count_tiles
from vqgan.display import console, tqdm


def parse_args(argv=None):
    data_defaults = DataConfig()
    train_defaults = VQGANTrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", default=data_defaults.index_path)
    parser.add_argument("--source", default="all", choices=("parquet", "folder", "all"))
    parser.add_argument("--tile-size", type=int, default=train_defaults.tile_size)
    parser.add_argument(
        "--tile-overlap-ratio", type=float, default=train_defaults.tile_overlap_ratio
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.index_path, encoding="utf-8") as f:
        index = json.load(f)

    for split in PARQUET_SPLITS:
        shards = build_shards(index, split, args.source)
        if not shards:
            continue

        n_images = n_small = n_tiles = 0
        pbar = tqdm(total=shard_total(shards), desc=f"{split:>10}", unit="img")
        for shard in shards:
            for w, h in iter_sizes(shard):
                n_images += 1
                pbar.update(1)
                tiles = count_tiles(h, w, args.tile_size, args.tile_overlap_ratio)
                if tiles == 0:
                    n_small += 1
                else:
                    n_tiles += tiles
        pbar.close()

        console.print(
            f"[bold]{split:>10}[/bold]: {n_images:,} image(s), {n_small:,} too small to crop at "
            f"{args.tile_size}px -> [bold]{n_tiles:,} crop(s)[/bold] "
            f"(overlap {args.tile_overlap_ratio})"
        )


if __name__ == "__main__":
    main()
