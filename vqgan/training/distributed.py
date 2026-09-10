"""Multi-GPU (DDP) plumbing, and the no-op it collapses to on one GPU.

Nothing here activates unless the run was launched by `torchrun` (which is
what sets RANK/LOCAL_RANK/WORLD_SIZE). A plain `python scripts/train_vqgan.py`
never enters a process group, so it keeps the exact single-process behavior
it has always had — no NCCL dependency, no new failure mode.

The division of labour with DDP itself: DDP all-reduces *gradients* of
parameters that require them. Anything else that has to agree across ranks is
this project's own problem — see VectorQuantizer, whose EMA codebook is
updated by hand in forward() with requires_grad=False and so is invisible to
DDP's gradient sync.
"""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def setup_distributed():
    """-> (is_distributed, rank, local_rank, world_size).

    Returns (False, 0, 0, 1) whenever this is a plain single-process run, so
    every caller can use `rank`/`world_size` unconditionally and have the
    arithmetic degenerate to what it computed before DDP existed.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return False, 0, 0, 1

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, local_rank, world_size


def cleanup_distributed(is_distributed):
    if is_distributed:
        dist.destroy_process_group()


def unwrap(module):
    """The underlying module, whether or not DDP is wrapping it.

    DDP forwards __call__, parameters() and state_dict(), but *not* attribute
    access to submodules — `ddp_vqgan.decoder` is an AttributeError. Anything
    reaching into the module tree (decoder.last_layer(), saving a checkpoint
    in the plain single-GPU format) goes through here.
    """
    if isinstance(module, DistributedDataParallel):
        return module.module
    return module
