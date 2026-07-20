"""Aspect-ratio bucket table, nearest-bucket assignment, and a bucketed batch sampler.

Every training batch must contain images from a single bucket only (standard
"aspect ratio bucketing", as used in SDXL and similar systems) since token count
differs per bucket and a batch tensor needs a uniform shape.
"""
import math
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import Sampler

DOWNSAMPLE_FACTOR = 32


@dataclass(frozen=True)
class Bucket:
    """A fixed (width, height) target resolution, both multiples of 32."""

    name: str
    width: int
    height: int

    def __post_init__(self) -> None:
        assert self.width % DOWNSAMPLE_FACTOR == 0, f"bucket {self.name} width must be a multiple of 32"
        assert self.height % DOWNSAMPLE_FACTOR == 0, f"bucket {self.name} height must be a multiple of 32"

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def grid(self) -> tuple[int, int]:
        """(grid_height, grid_width) -- matches the encoder output's [B, C, H, W] axis order."""
        return (self.height // DOWNSAMPLE_FACTOR, self.width // DOWNSAMPLE_FACTOR)

    @property
    def num_tokens(self) -> int:
        grid_h, grid_w = self.grid
        return grid_h * grid_w


# Bucket set with max side 768px -> max grid 24x24 (5 downsampling stages = 32x).
DEFAULT_BUCKETS: tuple[Bucket, ...] = (
    Bucket("1:1", 768, 768),
    Bucket("4:5", 608, 768),
    Bucket("5:4", 768, 608),
    Bucket("4:3", 768, 576),
    Bucket("3:4", 576, 768),
    Bucket("3:2", 768, 512),
    Bucket("2:3", 512, 768),
    Bucket("16:9", 768, 448),
    Bucket("9:16", 448, 768),
)


def assign_bucket(image_width: int, image_height: int, buckets=DEFAULT_BUCKETS) -> Bucket:
    """Assigns an image to the bucket with the closest aspect ratio (log-ratio distance,
    so portrait and landscape mismatches are penalized symmetrically), not by absolute size."""
    target_log_ratio = math.log(image_width / image_height)
    return min(buckets, key=lambda b: abs(math.log(b.aspect_ratio) - target_log_ratio))


def steps_per_epoch(bucket_counts: dict[int, int], batch_size: int, accumulation_steps: int,
                     world_size: int = 1) -> int:
    """Number of weight updates one epoch yields, computed straight from dataset counts --
    no shuffling/construction needed, so this is cheap to call just to size total_steps at
    startup.

    bucket_counts: {bucket_id: number of dataset images assigned to that bucket}.
    Per .claude/mulit-vqgan.md, "Training Length -- Epoch-Based": batches are counted per
    bucket (drop_last -- a bucket's remainder images that don't fill a full batch are
    dropped, since a batch must be single-bucket) and summed, since batch_size/
    accumulation_steps are global but bucket dataset sizes differ. The accumulation-cycle
    division happens on that combined total, not per bucket, since accumulation windows
    freely span bucket boundaries (batch_size/accumulation_steps are uniform across
    buckets, so there's no correctness reason to keep them separate).
    """
    total_batches = sum(count // batch_size for count in bucket_counts.values())
    per_rank_batches = total_batches // world_size if world_size > 1 else total_batches
    return per_rank_batches // accumulation_steps


class BucketedBatchSampler(Sampler):
    """Yields batches of dataset indices where every index in a batch belongs to the
    same bucket. Bucket assignment per index is precomputed once (passed in via
    `bucket_ids`); this sampler only handles shuffling and batching.

    Deterministic given (seed, epoch, batch_offset), so training can resume mid-epoch:
    `batch_offset` skips that many already-consumed batches from the reproducible
    per-epoch batch order.

    For multi-GPU (DistributedDataParallel) training, pass `rank`/`world_size`: every
    rank builds the identical full batch order (same seed+epoch, deterministic) and
    then takes a disjoint `[rank::world_size]` slice, truncated so every rank gets
    exactly the same number of batches -- required because DDP's backward pass is a
    collective operation and all ranks must call it the same number of times per epoch.
    """

    def __init__(
        self,
        bucket_ids: list[int],
        batch_size: int,
        seed: int = 42,
        epoch: int = 0,
        batch_offset: int = 0,
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.bucket_ids = bucket_ids
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.batch_offset = batch_offset
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size

        self.indices_by_bucket: dict[int, list[int]] = defaultdict(list)
        for idx, bucket_id in enumerate(bucket_ids):
            self.indices_by_bucket[bucket_id].append(idx)

    def set_state(self, epoch: int, batch_offset: int) -> None:
        self.epoch = epoch
        self.batch_offset = batch_offset

    def _build_all_batches(self) -> list[list[int]]:
        """The full, un-sharded batch order -- identical on every rank."""
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []

        for indices in self.indices_by_bucket.values():
            perm = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in perm]
            step = self.batch_size
            for start in range(0, len(shuffled), step):
                batch = shuffled[start : start + step]
                if len(batch) == step or not self.drop_last:
                    batches.append(batch)

        # shuffle batch order so buckets interleave through an epoch, not one after another
        batch_perm = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[i] for i in batch_perm]

    def _build_batches(self) -> list[list[int]]:
        all_batches = self._build_all_batches()
        if self.world_size == 1:
            return all_batches
        per_rank_count = len(all_batches) // self.world_size  # equal count on every rank
        return all_batches[self.rank :: self.world_size][:per_rank_count]

    def __iter__(self):
        batches = self._build_batches()
        yield from batches[self.batch_offset :]
        self.batch_offset = 0  # subsequent epochs start from the beginning

    def __len__(self) -> int:
        return len(self._build_batches()) - self.batch_offset
