"""Precompute and cache the in-training validation eval subset.

scripts/train_vqgan.py normally rebuilds this subset at the start of every
run: two header-only passes over the whole validation split to bucket images
by size, then decoding the crops it picks (see vqgan/data/eval_subset.py).
On ImageNet that scan takes real wall-clock time before the first training
step, and it is fully deterministic given the same
source/tile_size/tile_overlap_ratio/eval_images/eval_size_groups/seed — so a
run that will use those same settings can build it once here and load the
decoded result back instead of re-scanning.

Point VQGANTrainConfig.eval_prep_file (train_vqgan.py's --eval-prep-file) at
the file this writes. train_vqgan.py checks the cached settings against its
own config and refuses a mismatched cache rather than silently evaluating on
the wrong subset.

Usage:
    python scripts/prep_eval_data.py
    python scripts/prep_eval_data.py --source parquet --batch-size 110 --out data/eval_prep.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_vqgan import describe_shards, load_index
from vqgan.config import VQGANTrainConfig
from vqgan.data import build_shards
from vqgan.data.eval_subset import (
    build_balanced_val_subset,
    compute_log_size_edges,
    eval_image_floor,
    materialize_selected_crops,
)
from vqgan.display import console


def parse_args(argv=None):
    defaults = VQGANTrainConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index-path", default=defaults.index_path)
    parser.add_argument("--source", default=defaults.source, choices=("parquet", "folder", "all"))
    parser.add_argument("--tile-size", type=int, default=defaults.tile_size)
    parser.add_argument("--tile-overlap-ratio", type=float, default=defaults.tile_overlap_ratio)
    parser.add_argument("--eval-images", type=int, default=defaults.eval_images)
    parser.add_argument("--eval-size-groups", type=int, default=defaults.eval_size_groups)
    # Only used to reproduce eval_image_floor()'s batch-size-scaled minimum —
    # nothing here trains at this batch size.
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--out", default="data/eval_prep.pt")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    index = load_index(args.index_path)
    val_shards = build_shards(index, "validation", args.source)
    if not val_shards:
        raise RuntimeError(f"no validation shards for source={args.source!r} — check {args.index_path}")
    console.print(f"[bold]validation[/bold] {describe_shards(val_shards)}")

    eval_images = eval_image_floor(args.eval_images, args.batch_size)
    size_edges = compute_log_size_edges(
        val_shards, tile_size=args.tile_size, num_groups=args.eval_size_groups
    )
    selection = build_balanced_val_subset(
        val_shards, tile_size=args.tile_size, tile_overlap_ratio=args.tile_overlap_ratio,
        edges=size_edges, max_crops_target=eval_images, seed=args.seed,
    )
    crops, offsets = materialize_selected_crops(
        val_shards, selection, tile_size=args.tile_size, tile_overlap_ratio=args.tile_overlap_ratio,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "crops": crops,
        "offsets": offsets,
        "selection": selection,
        "params": {
            "source": args.source, "tile_size": args.tile_size,
            "tile_overlap_ratio": args.tile_overlap_ratio,
            "eval_images": eval_images, "eval_size_groups": args.eval_size_groups,
            "seed": args.seed,
        },
    }, out_path)
    console.print(
        f"[green]wrote[/green] {out_path} ({crops.shape[0]:,} crops from {len(selection):,} images)"
    )


if __name__ == "__main__":
    main()
