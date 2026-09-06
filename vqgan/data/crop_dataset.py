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

    `crops_per_image` amortizes the JPEG decode, which is the real per-image
    cost here: taking 2 crops out of one decoded image is nearly free compared
    to decoding a second image for the second crop.
    """

    def __init__(
        self,
        shards,
        *,
        tile_size,
        crops_per_image=2,
        shuffle_buffer=1024,
        augment=True,
        seed=0,
    ):
        self.shards = list(shards)
        self.tile_size = tile_size
        self.crops_per_image = crops_per_image
        self.shuffle_buffer = shuffle_buffer
        self.augment = augment
        self.seed = seed

    def _crops(self, image, rng):
        w, h = image.size
        tile = self.tile_size
        if w < tile or h < tile:
            return  # too small to crop at native resolution, and we never upscale
        for _ in range(self.crops_per_image):
            left = rng.randint(0, w - tile)
            top = rng.randint(0, h - tile)
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


def build_val_batch(shards, *, tile_size, num_images, seed=0) -> torch.Tensor:
    """A fixed [N, 3, tile, tile] batch of center crops, held in RAM.

    Validation has to be the *same* pixels every time for val L1 and FID to be
    comparable across steps, and it is small enough (512 crops ~= 100MB) to
    keep in memory rather than caching to disk — which also keeps the promise
    that this pipeline never writes image data back out.
    """
    crops = []
    for shard in shards:
        for image in shard.iter_images():
            w, h = image.size
            if w < tile_size or h < tile_size:
                continue
            left = (w - tile_size) // 2
            top = (h - tile_size) // 2
            crops.append(_to_tensor(image.crop((left, top, left + tile_size, top + tile_size))))
            if len(crops) >= num_images:
                return torch.stack(crops)
    if not crops:
        raise RuntimeError("no validation images were large enough to crop")
    return torch.stack(crops)
