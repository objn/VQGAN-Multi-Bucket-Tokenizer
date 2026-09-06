"""Streaming crop dataset.

Images are never resized. A training example is a `tile_size` x `tile_size`
window cut out of a source image at its native resolution, which is the same
thing `scripts/reconstruct.py` feeds the model at inference time — so what the
model trains on and what it is later asked to encode are the same distribution
of detail.

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

from .tiling import plan_tiles


def _to_tensor(crop) -> torch.Tensor:
    """PIL RGB crop -> CHW float tensor in [-1, 1]."""
    # np.array, not np.asarray: PIL hands back a read-only view, and
    # torch.from_numpy on a non-writable buffer warns on every call.
    array = np.array(crop, dtype=np.uint8)
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

    Every image is cut into the *complete* grid of tiles that `plan_tiles`
    lays out — the same planner scripts/reconstruct.py uses to feed a
    full-resolution photo through the model — so the crops the model trains
    on are drawn from exactly the distribution it will be asked to encode at
    inference, edge tiles and all. No cap on tiles per image: a large photo
    simply contributes more of them.

    An image that yields fewer than two tiles (only possible when it is
    exactly tile_size on both sides) has nothing to enumerate, so it falls
    back to a single in-bounds window instead.

    The same class serves train, validation and test: a split is simply the
    list of shards handed to it. Pass `shuffle=False, augment=False` for
    val/test and the pass becomes fully deterministic — same crops, same
    order, every time — which is what makes val L1 and FID comparable across
    evaluation points. There is deliberately no "how many" knob: a split is
    however many tiles its images contain.
    """

    def __init__(
        self,
        shards,
        *,
        tile_size,
        tile_overlap=0,
        shuffle=True,
        shuffle_buffer=1024,
        augment=True,
        seed=0,
    ):
        self.shards = list(shards)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.augment = augment
        self.seed = seed

    def _crops(self, image, rng):
        w, h = image.size
        tile = self.tile_size
        if w < tile or h < tile:
            return  # too small to crop at native resolution, and we never upscale

        origins = plan_tiles(h, w, tile, self.tile_overlap)
        if len(origins) < 2:
            # Nothing to enumerate. Training takes a random window; evaluation
            # takes the center one, so the pass stays reproducible.
            if self.shuffle:
                origins = [(rng.randint(0, h - tile), rng.randint(0, w - tile))]
            else:
                origins = [((h - tile) // 2, (w - tile) // 2)]

        for top, left in origins:
            crop = _to_tensor(image.crop((left, top, left + tile, top + tile)))
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
        # pass without the dataset having to track an epoch counter itself.
        rng = random.Random((torch.initial_seed() + self.seed) % (2**63))

        my_shards = self.shards[worker_id::num_workers]
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
