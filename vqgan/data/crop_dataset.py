"""Streaming crop dataset.

Images are never resized. A training example is a `tile_size` x `tile_size`
window cut out of a source image at its native resolution, the same size
`scripts/reconstruct.py` feeds the model at inference time — so what the model
trains on and what it is later asked to encode are the same distribution of
detail. Images smaller than a tile on either side are dropped rather than
upscaled.

This is an IterableDataset rather than a map-style one because the corpus is
1.28M JPEGs living inside 336 parquet shards: there is no cheap random access
to row N, but sequential reads are fast. Shards are handed out round-robin to
DataLoader workers and read front to back; randomness comes from shuffling the
shard order, sampling crop positions, and a reservoir shuffle buffer that mixes
together images that arrived close to each other in the file.
"""

import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms import functional as TF

from .tiling import jittered_tile_origins


def to_tensor(image) -> torch.Tensor:
    """PIL RGB image or crop -> CHW float tensor in [-1, 1]."""
    # np.array, not np.asarray: PIL hands back a read-only view, and
    # torch.from_numpy on a non-writable buffer warns on every call.
    array = np.array(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1.0


def random_white_balance(
    content: torch.Tensor, rng: random.Random, temp_strength=0.12, tint_strength=0.08
) -> torch.Tensor:
    """Random color-temperature (warm/cool: R vs. B) and tint (green vs.
    magenta: G vs. R+B) shift, mimicking a camera white-balance error —
    cheaper and more physically-motivated than generic per-channel jitter."""
    temp = rng.uniform(-temp_strength, temp_strength)  # + warmer, - cooler
    tint = rng.uniform(-tint_strength, tint_strength)  # + magenta, - green

    out = content.clone()
    out[0] = out[0] * (1.0 + temp)
    out[2] = out[2] * (1.0 - temp)
    out[1] = out[1] * (1.0 - tint)
    return out.clamp(-1.0, 1.0)


class CropDataset(IterableDataset):
    """Yields [3, tile_size, tile_size] tensors in [-1, 1], forever-ish.

    Every image is cut into the whole overlapping grid `jittered_tile_origins`
    lays out, so there is no cap on tiles per image: a large photo simply
    contributes more of them (an image only wide enough for one tile
    contributes exactly one). The grid is nudged by a few pixels on every
    pass, which means the same photo yields slightly different crops each
    epoch instead of the identical ones forever — position becomes an
    augmentation rather than a constant.

    The same class serves train, validation and test: a split is simply the
    list of shards handed to it. Pass `shuffle=False, augment=False` for
    val/test and the pass becomes fully deterministic — the jitter turns off
    with it, leaving a fixed centered grid — which is what makes val L1 and
    FID comparable across evaluation points. There is deliberately no "how
    many" knob: a split is however many tiles its images contain.
    """

    def __init__(
        self,
        shards,
        *,
        tile_size,
        tile_overlap_ratio=0.15,
        shuffle=True,
        shuffle_buffer=1024,
        augment=True,
        seed=0,
        rank=0,
        world_size=1,
    ):
        self.shards = list(shards)
        self.tile_size = tile_size
        self.tile_overlap_ratio = tile_overlap_ratio
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.augment = augment
        self.seed = seed
        # Which slice of the shard list this process takes, on top of the
        # per-worker split __iter__ already does. The defaults are the
        # single-process case and reduce that split to exactly what it was
        # before DDP existed — see __iter__.
        self.rank = rank
        self.world_size = world_size

    def _crops(self, image, rng):
        w, h = image.size
        tile = self.tile_size
        if w < tile or h < tile:
            return  # too small to crop at native resolution, and we never upscale

        # Jitter rides along with shuffling: on a val/test pass both are off and
        # the nth crop is always the same crop.
        origins = jittered_tile_origins(
            h, w, tile, self.tile_overlap_ratio, rng, jitter=self.shuffle
        )

        for top, left in origins:
            crop = to_tensor(image.crop((left, top, left + tile, top + tile)))
            if self.augment:
                if rng.random() < 0.5:
                    crop = TF.hflip(crop)
                crop = random_white_balance(crop, rng)
            yield crop

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        # torch.initial_seed() is re-derived by the DataLoader for every worker
        # on every epoch, so this reshuffles shards and crop positions each
        # pass without the dataset having to track an epoch counter itself. It
        # is identical across ranks, though — every process was seeded from the
        # same cfg.seed — so rank has to be mixed in separately, or two GPUs
        # would jitter and shuffle their (disjoint) shards in lockstep.
        rng = random.Random(
            (torch.initial_seed() + self.seed + self.rank * 100_003) % (2**63)
        )

        # Workers within a process and processes within the run are the same
        # kind of split, so they compose into one flat stride: worker w of rank
        # r takes every (world_size * num_workers)th shard. At rank=0,
        # world_size=1 this is exactly `shards[worker_id::num_workers]`.
        global_worker_id = self.rank * num_workers + worker_id
        total_workers = self.world_size * num_workers
        my_shards = self.shards[global_worker_id::total_workers]
        if not self.shuffle:
            # Deterministic pass: shards in index order, no reservoir, so the
            # nth crop is always the same crop.
            for shard in my_shards:
                for image in shard.iter_images():
                    yield from self._crops(image, rng)
            return

        rng.shuffle(my_shards)
        buffer = []
        for shard in my_shards:
            for image in shard.iter_images():
                for crop in self._crops(image, rng):
                    if len(buffer) < self.shuffle_buffer:
                        buffer.append(crop)
                        continue
                    i = rng.randrange(len(buffer))
                    buffer[i], crop = crop, buffer[i]
                    yield crop
        rng.shuffle(buffer)
        yield from buffer
