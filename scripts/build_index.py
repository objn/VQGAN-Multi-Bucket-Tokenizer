"""Stage 1: work out which images exist and which split each belongs to.

No pixels are touched and nothing is cached: this reads the parquet *footers*
(cheap — metadata only, not the 156GB of JPEG bytes) and lists the files under
`images/`, then writes a single data/index.json that the training dataset
streams from. That is the whole preprocessing step now — resizing, padding and
the pixels.npy memmap are gone, because the model crops at native resolution.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --folder-root images --out data/index.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqgan.config import DataConfig
from vqgan.data.sources import (
    PARQUET_SPLITS,
    count_parquet_rows,
    discover_images,
    parquet_files_for,
    read_manifests,
    verified_size,
)
from vqgan.display import console, tqdm


def parse_args(argv=None) -> DataConfig:
    defaults = DataConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default=defaults.manifest_dir)
    parser.add_argument("--folder-root", default=defaults.folder_root)
    parser.add_argument("--out", dest="index_path", default=defaults.index_path)
    parser.add_argument(
        "--min-size", type=int, default=defaults.min_size,
        help="drop images shorter than this on either side (default: the tile size)",
    )
    parser.add_argument("--val-frac", type=float, default=defaults.val_frac)
    parser.add_argument("--test-frac", type=float, default=defaults.test_frac)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args(argv)
    return DataConfig(**{**vars(defaults), **vars(args)})


def index_parquet(manifest_dir) -> dict:
    manifests = read_manifests(manifest_dir)
    if not manifests:
        console.print(f"[yellow]no *.json manifests in {manifest_dir}[/yellow] — parquet sources skipped")
        return {split: [] for split in PARQUET_SPLITS}

    index = {split: [] for split in PARQUET_SPLITS}
    for manifest in manifests:
        for split in PARQUET_SPLITS:
            files = parquet_files_for(manifest, split)
            index[split].extend(str(p) for p in files)
            n_images = count_parquet_rows(files)
            console.print(f"  {manifest['name']} / {split}: {len(files)} shard(s), {n_images:,} images")
    return index


def index_folder(cfg: DataConfig) -> dict:
    paths = discover_images(cfg.folder_root)
    if not paths:
        console.print(f"[yellow]no images under {cfg.folder_root}[/yellow]")
        return {split: [] for split in PARQUET_SPLITS}

    # Corrupt files and images too small to crop would each only surface as a
    # skipped image mid-training; catching them here means the index is a list
    # of things that actually get trained on, and the split counts below are
    # honest.
    usable, n_bad, n_small = [], 0, 0
    for path in tqdm(paths, desc="verifying"):
        size = verified_size(path)
        if size is None:
            n_bad += 1
        elif min(size) < cfg.min_size:
            n_small += 1
        else:
            usable.append(path)
    if n_bad:
        console.print(f"[yellow]skipped {n_bad} corrupt/unreadable file(s)[/yellow]")
    if n_small:
        console.print(f"[yellow]skipped {n_small} image(s) smaller than {cfg.min_size}px[/yellow]")
    if not usable:
        return {split: [] for split in PARQUET_SPLITS}

    rng = random.Random(cfg.seed)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * cfg.val_frac)
    n_test = round(len(shuffled) * cfg.test_frac)

    return {
        "validation": sorted(str(p) for p in shuffled[:n_val]),
        "test": sorted(str(p) for p in shuffled[n_val:n_val + n_test]),
        "train": sorted(str(p) for p in shuffled[n_val + n_test:]),
    }


def main(argv=None):
    cfg = parse_args(argv)

    console.print("[bold]parquet sources[/bold]")
    parquet = index_parquet(cfg.manifest_dir)

    console.print(f"[bold]folder source[/bold] ({cfg.folder_root})")
    folder = index_folder(cfg)

    index = {"parquet": parquet, "folder": folder}

    index_path = Path(cfg.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    for split in PARQUET_SPLITS:
        n_parquet_images = count_parquet_rows(parquet[split])
        console.print(
            f"{split:>10}: {len(parquet[split])} parquet shard(s) / {n_parquet_images:,} images"
            f", {len(folder[split])} folder image(s)"
        )
    console.print(f"[green]wrote[/green] {index_path}")


if __name__ == "__main__":
    main()
