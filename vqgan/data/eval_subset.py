"""A fixed, seeded, size-balanced subset of a validation split for the
in-training eval readout (scripts/train_vqgan.py's evaluate()).

A plain sequential (or random) prefix of `eval_images` crops is badly
skewed by native image size: ImageNet-1k validation is ~84% images that
crop into exactly 2 tiles (the ~500x375 "web-resized" convention), while
large/high-resolution images — worth far more crops each, and the images
the tiling grid actually stresses — are a rare, scattered tail. Sampling a
short prefix therefore almost never touches that tail.

`build_balanced_val_subset` fixes this: bucket every croppable validation
image by native (shorter-side) size into `num_groups` log-spaced groups,
then round-robin one uniformly random image per group, largest group first,
until `max_crops_target` crops have been claimed. It only reads image
headers (via `iter_sizes`), never pixels. `materialize_selected_crops` then
decodes exactly those images' pixels once, so a caller can cache the result
for a whole training run instead of re-scanning the split on every eval.
"""

import math
import random
from dataclasses import dataclass

import torch

from .crop_dataset import to_tensor
from .sources import iter_selected, iter_sizes
from .tiling import count_tiles, jittered_tile_origins


@dataclass
class SelectedImage:
    shard_index: int     # position in the `shards` list passed in
    index_in_shard: int  # position within shard.iter_images()/iter_sizes(shard)
    group_index: int     # 0 = smallest size group ... num_groups - 1 = largest
    num_crops: int       # count_tiles(h, w, tile_size, tile_overlap_ratio)
    take_crops: int      # <= num_crops; how many crops (raster order) are kept


def eval_image_floor(base_eval_images: int, batch_size: int) -> int:
    """The in-training eval readout's actual crop count: max(base_eval_images,
    64 * batch_size) — see VQGANTrainConfig.eval_images.

    Shared between scripts/train_vqgan.py (which applies it live) and
    scripts/prep_eval_data.py (which needs to reproduce the exact same number
    so a cache built ahead of time matches what a run would otherwise build
    for itself).
    """
    return max(base_eval_images, 64 * batch_size)


def compute_log_size_edges(shards, *, tile_size: int, num_groups: int) -> list[float]:
    """`num_groups` + 1 log-spaced breakpoints spanning the real observed
    shorter side of every croppable image in `shards` (header reads only).

    Computed fresh from the actual split rather than hardcoded, so it stays
    correct if the dataset changes. Degenerates to a single group (`[lo,
    lo]`) if every croppable image happens to share one exact size, since
    log-spacing a zero-width range is undefined.
    """
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")

    shorts = []
    for shard in shards:
        for w, h in iter_sizes(shard):
            if w < tile_size or h < tile_size:
                continue  # never croppable, regardless of overlap ratio
            shorts.append(min(w, h))
    if not shorts:
        raise ValueError("no croppable images found while computing size-group edges")

    lo, hi = min(shorts), max(shorts)
    if lo == hi:
        return [float(lo), float(lo)]

    log_lo, log_hi = math.log(lo), math.log(hi)
    edges = [math.exp(log_lo + (log_hi - log_lo) * i / num_groups) for i in range(num_groups + 1)]
    edges[0], edges[-1] = float(lo), float(hi)
    return edges


def _group_index(short: float, edges: list[float]) -> int:
    """Which of `len(edges) - 1` groups `short` falls into. The top group's
    upper edge is inclusive, so the largest image in the split lands in the
    top group instead of falling just outside every range."""
    last = len(edges) - 2
    for i in range(last + 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= short < hi or (i == last and lo <= short <= hi):
            return i
    return last  # short outside [edges[0], edges[-1]] — shouldn't happen; clamp to the top group


def build_balanced_val_subset(
    shards, *, tile_size: int, tile_overlap_ratio: float, edges: list[float],
    max_crops_target: int, seed: int,
) -> list[SelectedImage]:
    """Round-robin, size-balanced selection of crops for the eval readout.

    Buckets every croppable image (header reads only, via iter_sizes) into
    `len(edges) - 1` groups by shorter side, then repeatedly: skip any
    exhausted bucket immediately, pop a uniformly random image from the
    current bucket (largest-group-first, wrapping back to largest after the
    smallest), and take as many of its crops as still fit the quota. Stops
    the instant the quota is filled (the image that fills it may be only
    partially used) or every bucket has been emptied.

    `seed` drives every random pick directly (no offset/derivation) so the
    same `VQGANTrainConfig.seed` reproduces the exact same selection.
    """
    num_groups = len(edges) - 1
    buckets_by_group = [[] for _ in range(num_groups)]
    for shard_index, shard in enumerate(shards):
        for index_in_shard, (w, h) in enumerate(iter_sizes(shard)):
            num_crops = count_tiles(h, w, tile_size, tile_overlap_ratio)
            if num_crops == 0:
                continue
            group_index = _group_index(min(w, h), edges)
            buckets_by_group[group_index].append((shard_index, index_in_shard, num_crops))

    # Largest group first for round-robin; group_index tags stay the
    # original (0 = smallest) numbering regardless of this walk order.
    buckets = list(enumerate(buckets_by_group))[::-1]
    num_buckets = len(buckets)

    rng = random.Random(seed)
    selected: list[SelectedImage] = []
    total = 0
    bucket_pos = 0
    while total < max_crops_target:
        while not buckets[bucket_pos][1]:
            if all(not candidates for _, candidates in buckets):
                return selected
            bucket_pos = (bucket_pos + 1) % num_buckets

        group_index, candidates = buckets[bucket_pos]
        shard_index, index_in_shard, num_crops = candidates.pop(rng.randrange(len(candidates)))

        taking = min(num_crops, max_crops_target - total)
        selected.append(SelectedImage(
            shard_index=shard_index, index_in_shard=index_in_shard, group_index=group_index,
            num_crops=num_crops, take_crops=taking,
        ))
        total += taking
        if taking < num_crops:
            break  # quota filled exactly, mid-image — stop entirely

        bucket_pos = (bucket_pos + 1) % num_buckets

    return selected


def materialize_selected_crops(
    shards, selected: list[SelectedImage], *, tile_size: int, tile_overlap_ratio: float,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Decode `selected`'s crops once, as CPU float32 tensors in [-1, 1].

    Groups selections by shard so each shard containing at least one
    selected image is read sequentially exactly once (via iter_selected),
    not once per image. Crop positions use the same deterministic centered
    grid CropDataset uses for validation (jittered_tile_origins with
    jitter=False) — no augmentation, matching val's own augment=False.

    Returns the crops as one [N, 3, tile_size, tile_size] tensor, in blocks
    matching `selected`'s order, plus a parallel list of (start, end) row
    ranges — one per entry of `selected` — so a caller can slice out any one
    image's own crops (e.g. for a size-balanced recon-preview grid) without
    decoding anything twice.
    """
    by_shard: dict[int, dict[int, int]] = {}
    for position, sel in enumerate(selected):
        by_shard.setdefault(sel.shard_index, {})[sel.index_in_shard] = position

    crops_per_selection: list[list[torch.Tensor]] = [[] for _ in selected]
    no_jitter_rng = random.Random(0)  # jitter=False never draws from this
    for shard_index, wanted in by_shard.items():
        for index_in_shard, image in iter_selected(shards[shard_index], set(wanted)):
            position = wanted[index_in_shard]
            sel = selected[position]
            w, h = image.size
            origins = jittered_tile_origins(
                h, w, tile_size, tile_overlap_ratio, no_jitter_rng, jitter=False
            )[:sel.take_crops]
            crops_per_selection[position] = [
                to_tensor(image.crop((left, top, left + tile_size, top + tile_size)))
                for top, left in origins
            ]

    all_crops: list[torch.Tensor] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for crops in crops_per_selection:
        offsets.append((cursor, cursor + len(crops)))
        all_crops.extend(crops)
        cursor += len(crops)

    stacked = torch.stack(all_crops, dim=0) if all_crops else torch.empty(0, 3, tile_size, tile_size)
    return stacked, offsets
