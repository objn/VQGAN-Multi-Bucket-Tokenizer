"""Covers BucketedBatchSampler (single-bucket batches, deterministic resume, equal
DDP sharding) and steps_per_epoch() -- the epoch-based training-length calculation
(see .claude/mulit-vqgan.md, "Training Length -- Epoch-Based").
"""
from collections import Counter

from vqgan.data.buckets import Bucket, BucketedBatchSampler, steps_per_epoch

BUCKETS = [
    Bucket("big", 64, 64),
    Bucket("small", 32, 32),
]
BATCH_SIZE = 4
ACCUM_STEPS = 3


def make_bucket_ids(big_count: int, small_count: int) -> list[int]:
    return [0] * big_count + [1] * small_count


def test_steps_per_epoch_matches_formula():
    # 20 big // 4 = 5 batches, 17 small // 4 = 4 batches -> 9 batches // 3 accum = 3 steps
    counts = Counter(make_bucket_ids(20, 17))
    assert steps_per_epoch(dict(counts), BATCH_SIZE, ACCUM_STEPS) == 3


def test_steps_per_epoch_zero_when_not_enough_images():
    counts = Counter(make_bucket_ids(1, 1))  # not even one full batch
    assert steps_per_epoch(dict(counts), BATCH_SIZE, ACCUM_STEPS) == 0


def test_batches_are_single_bucket():
    bucket_ids = make_bucket_ids(20, 17)
    sampler = BucketedBatchSampler(bucket_ids, batch_size=BATCH_SIZE, seed=0)
    for batch in sampler:
        bucket_ids_in_batch = {bucket_ids[i] for i in batch}
        assert len(bucket_ids_in_batch) == 1, "a single batch must never mix buckets"
        assert len(batch) == BATCH_SIZE  # drop_last=True by default


def test_resume_batch_offset_skips_already_consumed_batches():
    bucket_ids = make_bucket_ids(20, 17)
    seed = 0
    full = list(BucketedBatchSampler(bucket_ids, batch_size=BATCH_SIZE, seed=seed))

    resumed = list(BucketedBatchSampler(bucket_ids, batch_size=BATCH_SIZE, seed=seed, batch_offset=3))
    assert resumed == full[3:]


def test_world_size_shards_equally_across_ranks():
    bucket_ids = make_bucket_ids(40, 34)
    seed = 0
    rank0 = list(BucketedBatchSampler(bucket_ids, batch_size=BATCH_SIZE, seed=seed, rank=0, world_size=2))
    rank1 = list(BucketedBatchSampler(bucket_ids, batch_size=BATCH_SIZE, seed=seed, rank=1, world_size=2))

    # equal batch count on every rank -- required since DDP's backward pass is a
    # collective operation, so every rank must call optimizer.step() the same number
    # of times per epoch
    assert len(rank0) == len(rank1)

    # no overlap between ranks' dataset indices
    rank0_indices = {i for batch in rank0 for i in batch}
    rank1_indices = {i for batch in rank1 for i in batch}
    assert rank0_indices.isdisjoint(rank1_indices)
