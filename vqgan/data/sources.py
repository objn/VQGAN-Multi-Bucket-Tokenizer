"""Where training images come from.

Two kinds of source, behind one interface (`iter_images()` yielding PIL RGB
images):

  - `ParquetShard` — one .parquet file of a HuggingFace image dataset
    (ImageNet-1k here), read straight off disk. The dataset is 156GB of JPEG
    bytes embedded in the parquet columns; decoding it to loose files first
    would double that on disk for no benefit, so shards are streamed instead.
  - `FolderShard` — a chunk of ordinary image files from `images/`, for
    dropping your own pictures in.

Both are cheap to construct and only touch the disk while being iterated, so a
DataLoader worker can be handed a list of them and start reading immediately.
"""

import io
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PARQUET_SPLITS = ("train", "validation", "test")


def read_manifests(manifest_dir) -> list[dict]:
    """Load the *.json pointers in `manifest_dir`.

    The parquet datasets live on a separate disk, so the repo only keeps a
    small JSON per dataset naming the directory to read from — nothing is
    copied into the project tree.
    """
    manifest_dir = Path(manifest_dir)
    if not manifest_dir.is_dir():
        return []
    manifests = []
    for path in sorted(manifest_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["name"] = path.stem
        manifests.append(manifest)
    return manifests


def parquet_files_for(manifest: dict, split: str) -> list[Path]:
    """The shards of one split, from a manifest entry.

    Split membership comes from the filename prefix HuggingFace writes
    (`train-00000-of-00294.parquet`), which is also what the dataset card's
    `data_files` globs match on.
    """
    root = manifest.get("parquet_files") or (Path(manifest["root"]) / "data")
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"parquet directory not found: {root}")
    return sorted(root.glob(f"{split}-*.parquet"))


def count_parquet_rows(paths) -> int:
    """Total row count across parquet files, from footer metadata only —
    each file's row count is stored alongside its schema, so this touches no
    image bytes and costs nothing close to reading the (156GB) dataset.

    Shard *file* counts are a poor proxy for dataset size (ImageNet-1k's
    294 train shards hold ~4,357 images each): this is what build_index.py
    reports instead, so "294 shards" doesn't read as "294 images."
    """
    import pyarrow.parquet as pq  # lazy: keep the package importable without pyarrow

    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)


def shard_total(shards) -> int:
    """Image count across shards from cheap metadata only — parquet row
    counts from footers, folder shards from their own path lists. No image
    bytes touched, so this is safe to call just to size a progress bar."""
    parquet_paths = [s.path for s in shards if isinstance(s, ParquetShard)]
    total = count_parquet_rows(parquet_paths) if parquet_paths else 0
    total += sum(len(s.paths) for s in shards if isinstance(s, FolderShard))
    return total


def iter_sizes(shard):
    """(width, height) of every image in a shard, decoding headers only —
    unlike iter_images(), never calls .convert(), which is what forces a
    full JPEG decode. Same skip semantics as iter_images() (a corrupt/
    unreadable record is silently skipped), so index positions here line up
    with iter_images()'s and iter_selected()'s for the same shard."""
    if isinstance(shard, ParquetShard):
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(shard.path)
        for batch in parquet_file.iter_batches(batch_size=shard.batch_size, columns=["image"]):
            for record in batch.column("image").to_pylist():
                try:
                    with Image.open(io.BytesIO(record["bytes"])) as im:
                        yield im.size
                except (UnidentifiedImageError, OSError, ValueError):
                    continue
    else:
        for path in shard.paths:
            try:
                with Image.open(path) as im:
                    yield im.size
            except (UnidentifiedImageError, OSError, ValueError):
                continue


def iter_selected(shard, indices: set[int]):
    """Yield (index, PIL RGB image) for the 0-based positions in `indices`,
    in the same order iter_images()/iter_sizes() would visit them — `index`
    counts only records that pass the same open-and-read-size check
    iter_sizes() uses, in the same order, so a shard with a corrupt/
    unreadable record in it still numbers identically under both functions
    (neither counts that record; both silently skip it).

    For a parquet shard there is no random row access (see ParquetShard's
    docstring), so this still reads every row sequentially — but it only
    pays the full Image.open(...).convert("RGB") decode cost for rows whose
    index is in `indices`; everything else only pays iter_sizes()'s cheap
    header-read cost. A folder shard's paths are already a plain indexable
    list, so this opens exactly the wanted files (plus the same cheap
    validity check, to keep its numbering consistent with iter_sizes() too)."""
    if isinstance(shard, ParquetShard):
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(shard.path)
        i = 0
        for batch in parquet_file.iter_batches(batch_size=shard.batch_size, columns=["image"]):
            for record in batch.column("image").to_pylist():
                raw = record["bytes"]
                try:
                    with Image.open(io.BytesIO(raw)) as im:
                        im.size
                except (UnidentifiedImageError, OSError, ValueError):
                    continue
                if i in indices:
                    try:
                        with Image.open(io.BytesIO(raw)) as im:
                            yield i, im.convert("RGB")
                    except (UnidentifiedImageError, OSError, ValueError):
                        pass
                i += 1
    else:
        i = 0
        for path in shard.paths:
            try:
                with Image.open(path) as im:
                    im.size
            except (UnidentifiedImageError, OSError, ValueError):
                continue
            if i in indices:
                try:
                    with Image.open(path) as im:
                        yield i, im.convert("RGB")
                except (UnidentifiedImageError, OSError, ValueError):
                    pass
            i += 1


def discover_images(root) -> list[Path]:
    """Every image file under `root`, recursively, in a stable order."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def verified_size(path):
    """(width, height) of an image that opens cleanly, else None.

    Reads the header only — `Image.open` is lazy and `verify()` checks the
    file's integrity without decoding it — so this stays cheap enough to run
    over every file in `images/` while building the index. `size` has to be
    read *before* `verify()`, which leaves the image object unusable.
    """
    try:
        with Image.open(path) as im:
            size = im.size
            im.verify()
        return size
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def is_valid_image(path) -> bool:
    """Open + verify an image is not corrupt/truncated."""
    return verified_size(path) is not None


@dataclass
class ParquetShard:
    """One parquet file, streamed row by row.

    Unlike the folder source, images too small to crop cannot be filtered out
    while building the index: parquet footers carry row counts and schemas, not
    image dimensions, so learning a row's size means decoding its JPEG. They are
    dropped by CropDataset at training time instead.
    """

    path: str
    batch_size: int = 64

    def iter_images(self):
        # Imported lazily so the rest of the package stays importable without
        # pyarrow (visualize_model, reconstruct on loose files, ...).
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(self.path)
        # iter_batches, not read_row_group: a row group inside these ~500MB
        # shards is far too big to materialize per DataLoader worker, and we
        # only ever walk it forwards anyway.
        for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=["image"]):
            for record in batch.column("image").to_pylist():
                try:
                    with Image.open(io.BytesIO(record["bytes"])) as im:
                        yield im.convert("RGB")
                except (UnidentifiedImageError, OSError, ValueError):
                    continue


@dataclass
class FolderShard:
    paths: list

    def iter_images(self):
        for path in self.paths:
            try:
                with Image.open(path) as im:
                    yield im.convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError):
                continue


SOURCES = ("parquet", "folder", "all")


def build_shards(index: dict, split: str, source: str = "all", *, folder_chunk: int = 64) -> list:
    """Turn one split of a `data/index.json` into a flat list of shards.

    `source` picks which corpus the split is drawn from, and the two are kept
    apart on purpose because their splits mean different things:

      - "parquet" is the downloaded dataset, which ships its own
        train/validation/test division — pretraining uses it as-is.
      - "folder" is whatever the user dropped in images/, split by the
        val_frac/test_frac ratios they chose when building the index —
        finetuning uses that.

    Mixing them would silently pretrain on the user's finetuning data and
    make the two sets of split semantics indistinguishable.

    Folder images are grouped into chunks so that a run of them is a
    comparable unit of work to a parquet shard, which keeps the round-robin
    split across DataLoader workers roughly balanced.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")

    shards = []
    if source in ("parquet", "all"):
        shards += [ParquetShard(path=p) for p in index["parquet"].get(split, [])]
    if source in ("folder", "all"):
        folder_paths = index["folder"].get(split, [])
        for i in range(0, len(folder_paths), folder_chunk):
            shards.append(FolderShard(paths=folder_paths[i:i + folder_chunk]))
    return shards
