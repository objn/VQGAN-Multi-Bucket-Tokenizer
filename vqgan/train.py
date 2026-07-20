"""Main training entrypoint for the VQGAN tokenizer.

Single-GPU usage:
    uv run python -m vqgan.train --train-dir data/train --val-dir data/val

Multi-GPU (DistributedDataParallel) usage, one process per GPU via torchrun:
    torchrun --standalone --nproc_per_node=2 -m vqgan.train --train-dir data/train --val-dir data/val
"""
import argparse
import copy
import dataclasses
import os
import random
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vqgan.config import Config, DEFAULT_CONFIG_PATH, load_config
from vqgan.data.buckets import Bucket, BucketedBatchSampler, steps_per_epoch as compute_steps_per_epoch
from vqgan.data.dataset import BucketedImageDataset
from vqgan.losses.losses import VQGANLoss
from vqgan.models.discriminator import PatchGANDiscriminator
from vqgan.models.vqgan import VQGAN

# rolling window size for step-speed/ETA smoothing -- large enough that the first few
# steps' CUDA warmup / cuDNN autotune overhead doesn't dominate the estimate
STEP_TIME_WINDOW = 50
STEP_TIME_SPIKE_FACTOR = 5.0  # console warning threshold: current step vs rolling avg

# config.model fields that determine checkpointed tensor shapes / param grouping --
# resuming with any of these changed would silently corrupt training (or fail with a
# confusing low-level shape-mismatch error deep inside load_state_dict), so they're
# checked explicitly with a clear message before that happens. Everything else in
# ModelConfig (commitment_beta, ema_decay, dead_code_revival, codebook_init, ...) is
# safe to change across a resume.
ARCHITECTURE_CRITICAL_MODEL_FIELDS = (
    "in_channels", "base_channels", "channel_mults", "latent_channels", "num_res_blocks",
    "codebook_size", "codebook_dim", "discriminator_channels", "discriminator_num_layers", "use_ema",
)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_distributed_info() -> tuple[int, int, int]:
    """(local_rank, rank, world_size), read from the env vars torchrun sets on every
    spawned process. All zero/one (i.e. single-process) if not launched via torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return local_rank, rank, world_size


def parse_args() -> tuple[Config, bool, str]:
    """All hyperparameters (including the bucket table) come from a single JSON file
    (--config, default configs/vqgan-multi.json). CLI flags below are optional
    overrides on top of it -- only flags actually passed on the command line take
    effect (default=None). Returns (config, fresh, resolved_config_path) -- the path
    is needed by the caller for the Config File Sync write-back.
    """
    parser = argparse.ArgumentParser(description="Train the VQGAN multi-resolution bucket tokenizer")

    parser.add_argument("--config", type=str, default=None, help=f"path to a JSON config (default: {DEFAULT_CONFIG_PATH})")

    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--target-epochs", type=int, default=None,
        help="how many full passes over the training dataset to train for; total_steps is "
             "derived from this at startup, not set directly",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="physical batch size, per GPU")
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--lr-generator", type=float, default=None)
    parser.add_argument("--lr-discriminator", type=float, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true", default=None)
    parser.add_argument("--no-grad-checkpointing", action="store_true", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument(
        "--fresh", action="store_true", default=False,
        help="start from step 0 even if output_dir/checkpoints/latest.pt exists (default: auto-resume from it)",
    )

    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--use-ema", action="store_true", default=None)
    parser.add_argument("--discriminator-start-step", type=int, default=None)
    parser.add_argument("--lambda-perceptual", type=float, default=None)
    parser.add_argument("--lambda-adv", type=float, default=None)
    parser.add_argument("--commitment-beta", type=float, default=None)
    parser.add_argument("--dead-code-revival", action="store_true", default=None)
    parser.add_argument("--dead-code-revival-every", type=int, default=None)
    parser.add_argument("--codebook-init", type=str, default=None, choices=["uniform", "kmeans"])
    parser.add_argument("--kmeans-init-batches", type=int, default=None)
    parser.add_argument("--kmeans-init-iters", type=int, default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)

    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--image-log-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)

    args = parser.parse_args()
    config = load_config(args.config)
    resolved_config_path = args.config if args.config else str(DEFAULT_CONFIG_PATH)

    if args.train_dir is not None:
        config.data.train_dir = args.train_dir
    if args.val_dir is not None:
        config.data.val_dir = args.val_dir
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
        config.train.num_workers = args.num_workers

    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    if args.target_epochs is not None:
        config.train.target_epochs = args.target_epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.accumulation_steps is not None:
        config.train.accumulation_steps = args.accumulation_steps
    if args.lr_generator is not None:
        config.train.lr_generator = args.lr_generator
    if args.lr_discriminator is not None:
        config.train.lr_discriminator = args.lr_discriminator
    if args.no_mixed_precision is not None:
        config.train.mixed_precision = not args.no_mixed_precision
    if args.no_grad_checkpointing is not None:
        config.train.grad_checkpointing = not args.no_grad_checkpointing
    if args.seed is not None:
        config.train.seed = args.seed
    if args.resume_from is not None:
        config.train.resume_from = args.resume_from
    if args.log_every is not None:
        config.train.log_every = args.log_every
    if args.image_log_every is not None:
        config.train.image_log_every = args.image_log_every
    if args.checkpoint_every is not None:
        config.train.checkpoint_every = args.checkpoint_every
    if args.val_every is not None:
        config.train.val_every = args.val_every
    if args.dead_code_revival_every is not None:
        config.train.dead_code_revival_every = args.dead_code_revival_every
    if args.kmeans_init_batches is not None:
        config.train.kmeans_init_batches = args.kmeans_init_batches
    if args.kmeans_init_iters is not None:
        config.train.kmeans_init_iters = args.kmeans_init_iters
    if args.lr_warmup_steps is not None:
        config.train.lr_warmup_steps = args.lr_warmup_steps

    if args.codebook_size is not None:
        config.model.codebook_size = args.codebook_size
    if args.use_ema is not None:
        config.model.use_ema = args.use_ema
    if args.discriminator_start_step is not None:
        config.loss.discriminator_start_step = args.discriminator_start_step
    if args.lambda_perceptual is not None:
        config.loss.lambda_perceptual = args.lambda_perceptual
    if args.lambda_adv is not None:
        config.loss.lambda_adv = args.lambda_adv
    if args.commitment_beta is not None:
        config.model.commitment_beta = args.commitment_beta
    if args.dead_code_revival is not None:
        config.model.dead_code_revival = args.dead_code_revival
    if args.codebook_init is not None:
        config.model.codebook_init = args.codebook_init

    return config, args.fresh, resolved_config_path


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for TensorBoard image logging."""
    return (x.clamp(-1, 1) + 1) / 2


def sync_config_to_disk(config: Config, config_path: str) -> None:
    """Writes the config back to disk if the resolved (post-CLI-override, post-derived-
    total_steps) values differ from what's currently on disk -- configs/vqgan-multi.json
    must always reflect the hyperparameters actually in use, not just what was there at
    launch (see .claude/mulit-vqgan.md, "Config File Sync"). Atomic write via
    Config.to_json(). `resume_from` is excluded from the comparison/write: it's transient
    per-invocation state (e.g. auto-detected latest.pt), not a persisted hyperparameter.
    """
    to_write = copy.deepcopy(config)
    to_write.train.resume_from = ""

    p = Path(config_path)
    on_disk = Config.from_json(str(p)) if p.exists() else None
    if on_disk is not None:
        on_disk = copy.deepcopy(on_disk)
        on_disk.train.resume_from = ""
    if on_disk is None or on_disk.to_dict() != to_write.to_dict():
        to_write.to_json(str(p))
        print(f"Config file updated: {p}")


def check_resume_config_compatible(ckpt_config: dict, config: Config) -> None:
    """Fails loudly (before load_state_dict gets a chance to produce a confusing
    shape-mismatch error, or worse, silently succeed with mismatched semantics) if any
    architecture-critical field differs between the checkpoint and the current config.
    See .claude/mulit-vqgan.md, Checkpointing: "resuming with a mismatched config fails
    loudly instead of silently corrupting training."
    """
    ckpt_model = ckpt_config.get("model", {})
    current_model = dataclasses.asdict(config.model)
    mismatches = {
        field: (ckpt_model.get(field), current_model.get(field))
        for field in ARCHITECTURE_CRITICAL_MODEL_FIELDS
        if ckpt_model.get(field) != current_model.get(field)
    }
    if mismatches:
        lines = "\n".join(f"  {k}: checkpoint={v[0]!r} vs current config={v[1]!r}" for k, v in mismatches.items())
        raise RuntimeError(
            "Refusing to resume: checkpoint's model config differs from the current config "
            f"in architecture-critical field(s):\n{lines}\n"
            "Resuming with a mismatched architecture would silently corrupt training (or fail "
            "with a confusing internal shape error). Fix the config to match, or start fresh with --fresh."
        )


def save_checkpoint(path: Path, step: int, epoch: int, batch_offset: int, raw_model, raw_discriminator,
                     opt_g, opt_d, scheduler_g, scheduler_d, config: Config) -> None:
    """Takes the raw (un-DDP-wrapped) model/discriminator so checkpoints have clean
    state_dict keys (no "module." prefix) and load identically in single-GPU or
    multi-GPU runs, and in eval.py (which always builds a plain VQGAN).

    Writes to a temp file and renames into place afterward -- a crash or Ctrl+C mid-write
    must never leave a corrupted file sitting at `path` for the resume logic to trip over.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "batch_offset": batch_offset,
            "model": raw_model.state_dict(),
            "discriminator": raw_discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "scheduler_g": scheduler_g.state_dict(),
            "scheduler_d": scheduler_d.state_dict(),
            "config": dataclasses.asdict(config),
            # for bit-identical resume of the data ordering / augmentation stream
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        tmp_path,
    )
    os.replace(tmp_path, path)  # atomic on both POSIX and Windows (NTFS)


def prune_old_checkpoints(ckpt_dir: Path, keep_last_n: int, milestone_every: int) -> None:
    """Keeps a rolling window of the most recent `keep_last_n` step_*.pt checkpoints, plus
    any that land on a permanent milestone (step % milestone_every == 0), and deletes the
    rest. Never touches latest.pt. Protects disk space without losing the ability to roll
    back to an old, known-good milestone if a recent checkpoint turns out to be corrupted
    or a training run regresses."""
    step_ckpts = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.removeprefix("step_")),
    )
    recent = set(step_ckpts[-keep_last_n:]) if keep_last_n > 0 else set()
    for p in step_ckpts:
        step = int(p.stem.removeprefix("step_"))
        is_milestone = milestone_every > 0 and step % milestone_every == 0
        if p not in recent and not is_milestone:
            p.unlink()


def run_kmeans_init(raw_model: VQGAN, train_dataset: BucketedImageDataset,
                     config: Config, device, is_distributed: bool, is_main: bool) -> None:
    """Seeds the codebook with k-means centroids over encoder output from a handful of initial
    batches, instead of the tiny-range uniform init -- run once before training starts, only when
    config.model.codebook_init == "kmeans". Falls back to the existing uniform init (no-op) if
    fewer than codebook_size vectors were collected.
    """
    if is_main:
        tqdm.write("Running k-means codebook init...")

    # deliberately single-shard (rank=0, world_size=1) even under DDP -- every rank runs this
    # independently from its own local data; see the DDP-desync warning below
    kmeans_sampler = BucketedBatchSampler(
        train_dataset.bucket_ids, batch_size=config.train.batch_size, seed=config.train.seed, rank=0, world_size=1,
    )
    loader = DataLoader(train_dataset, batch_sampler=kmeans_sampler, num_workers=config.train.num_workers)

    raw_model.encoder.eval()
    collected = []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= config.train.kmeans_init_batches:
                break
            images = images.to(device)
            z = raw_model.encoder(images)  # [B, C, H, W]
            z_flat = z.permute(0, 2, 3, 1).reshape(-1, z.size(1))  # [B*H*W, C], matches quantizer's own flattening
            collected.append(z_flat.cpu())
    raw_model.encoder.train()

    all_vectors = torch.cat(collected, dim=0)  # [N, C]
    n = all_vectors.size(0)
    k = raw_model.quantizer.codebook_size

    if n < k:
        if is_main:
            tqdm.write(
                f"WARNING: k-means init collected only {n} encoder-output vectors but codebook_size={k} "
                f"-- falling back to uniform init (increase train.kmeans_init_batches to collect >= {k} vectors)"
            )
        return  # embedding.weight already has its uniform init from the VectorQuantizer constructor

    if is_distributed and is_main:
        tqdm.write(
            "WARNING: k-means codebook init writes directly to embedding.weight.data outside the "
            "optimizer/autograd path and is NOT broadcast to other DDP ranks -- each rank runs this "
            "independently from its own local data and may end up with a different codebook. Not "
            "fixed this round (single-GPU correctness only); a future change should dist.broadcast "
            "rank 0's centroids to all ranks."
        )

    # from-scratch Lloyd's-algorithm k-means in torch, no new dependency
    all_vectors = all_vectors.to(device)
    init_idx = torch.randperm(n, device=device)[:k]
    centroids = all_vectors[init_idx].clone()  # [k, C]

    chunk = 4096
    for _ in range(config.train.kmeans_init_iters):
        # assign step, chunked to bound peak memory against a large N x k distance matrix
        assignments = torch.empty(n, dtype=torch.long, device=device)
        for start in range(0, n, chunk):
            v = all_vectors[start:start + chunk]  # [c, C]
            d = v.pow(2).sum(1, keepdim=True) - 2 * v @ centroids.t() + centroids.pow(2).sum(1)
            assignments[start:start + chunk] = d.argmin(dim=1)

        # update step
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=device)
        new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand(-1, centroids.size(1)), all_vectors)
        counts.scatter_add_(0, assignments, torch.ones(n, device=device))
        empty = counts == 0
        counts = counts.clamp(min=1)
        new_centroids = new_centroids / counts.unsqueeze(1)
        new_centroids[empty] = centroids[empty]  # keep empty clusters' previous centroid, don't collapse to zero
        centroids = new_centroids

    raw_model.quantizer.embedding.weight.data.copy_(centroids)
    if raw_model.quantizer.use_ema:
        raw_model.quantizer.ema_embed_avg.copy_(centroids)
        raw_model.quantizer.ema_cluster_size.fill_(1.0)

    if is_main:
        tqdm.write(f"K-means init done: {n} vectors -> {k} centroids ({config.train.kmeans_init_iters} iterations)")


def main() -> None:
    config, fresh, config_path = parse_args()
    torch.manual_seed(config.train.seed)  # must be identical across ranks: BucketedBatchSampler's
    # deterministic full-batch-order-then-shard scheme depends on every rank computing the same order
    buckets = [Bucket(**b) for b in config.data.buckets]

    local_rank, rank, world_size = get_distributed_info()
    is_distributed = world_size > 1
    is_main = rank == 0

    if is_distributed:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"

    # resume is the default, safe path: auto-detect latest.pt in this run's output_dir unless
    # the caller passed an explicit --resume-from or --fresh. Running the same command twice
    # must not silently restart from epoch/step 0.
    if not config.train.resume_from and not fresh:
        auto_latest = ckpt_dir / "latest.pt"
        if auto_latest.exists():
            config.train.resume_from = str(auto_latest)
            if is_main:
                print(f"Auto-detected {auto_latest}, resuming (pass --fresh to start over instead)")

    # raw_model/raw_discriminator always refer to the plain nn.Module -- used for
    # .encoder/.decoder/.quantizer access, state_dict()/load_state_dict(), and
    # validation. `model`/`discriminator` get reassigned to the DDP wrapper below and
    # are used only for the actual training forward+backward calls (so gradients sync
    # across ranks); DDP does NOT transparently proxy custom submodule access like
    # `.encoder`, only .parameters()/.state_dict()/etc, and state_dict() on a DDP
    # wrapper adds a "module." key prefix that would break single-GPU checkpoint compat.
    raw_model = VQGAN(config.model, grad_checkpointing=config.train.grad_checkpointing).to(device)
    raw_discriminator = PatchGANDiscriminator(
        in_channels=config.model.in_channels,
        base_channels=config.model.discriminator_channels,
        num_layers=config.model.discriminator_num_layers,
    ).to(device)
    criterion = VQGANLoss(config.loss).to(device)

    train_dataset = BucketedImageDataset(
        config.data.train_dir, buckets=buckets, horizontal_flip=config.data.horizontal_flip, train=True
    )
    val_dataset = None
    if Path(config.data.val_dir).exists():
        val_dataset = BucketedImageDataset(config.data.val_dir, buckets=buckets, train=False)

    # Training Length -- Epoch-Based: 1 epoch = the model has seen every training image
    # once, regardless of batch_size. steps_per_epoch is computed per bucket (dataset
    # count for that bucket, drop_last at the batch level) and summed, since bucket
    # dataset sizes differ even though batch_size/accumulation_steps are global -- NOT
    # a single dataset_size / batch_size division. This is recomputed fresh every run
    # from the live dataset, never hardcoded or cached.
    bucket_counts = Counter(train_dataset.bucket_ids)
    steps_per_epoch = compute_steps_per_epoch(
        dict(bucket_counts), config.train.batch_size, config.train.accumulation_steps, world_size=world_size
    )
    if steps_per_epoch == 0:
        raise RuntimeError(
            "steps_per_epoch computed as 0 -- not enough images to fill even one full "
            "batch_size*accumulation_steps cycle. Lower train.batch_size/accumulation_steps "
            "in the config, or add more training data."
        )
    config.train.total_steps = config.train.target_epochs * steps_per_epoch

    if config.model.codebook_init == "kmeans" and not config.train.resume_from:
        run_kmeans_init(raw_model, train_dataset, config, device, is_distributed, is_main)

    if is_main:
        sync_config_to_disk(config, config_path)

    generator_params = list(raw_model.encoder.parameters()) + list(raw_model.decoder.parameters())
    if not config.model.use_ema:
        generator_params += list(raw_model.quantizer.parameters())
    opt_g = torch.optim.AdamW(
        generator_params, lr=config.train.lr_generator, betas=(config.train.beta1, config.train.beta2)
    )
    opt_d = torch.optim.AdamW(
        raw_discriminator.parameters(),
        lr=config.train.lr_discriminator,
        betas=(config.train.beta1, config.train.beta2),
    )

    def warmup_lr_lambda(current_step: int) -> float:
        warmup_steps = config.train.lr_warmup_steps
        return 1.0 if warmup_steps <= 0 else min(1.0, (current_step + 1) / warmup_steps)

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=warmup_lr_lambda)
    # opt_d.step() only starts firing once the discriminator activates (discriminator_start_step),
    # so scheduler_d.step() (called from that same point on) naturally gives it its own independent
    # post-activation warmup window with no extra bookkeeping needed
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=warmup_lr_lambda)

    sampler = BucketedBatchSampler(
        train_dataset.bucket_ids, batch_size=config.train.batch_size, seed=config.train.seed,
        rank=rank, world_size=world_size,
    )

    step = 0
    epoch = 0

    if config.train.resume_from:
        # weights_only=False: our own checkpoints also carry numpy/python RNG state (not just
        # tensors), which torch.load's default weights_only=True (PyTorch >=2.6) rejects. Safe
        # here since these checkpoints are self-produced, not loaded from an untrusted source.
        ckpt = torch.load(config.train.resume_from, map_location=device, weights_only=False)
        check_resume_config_compatible(ckpt["config"], config)
        raw_model.load_state_dict(ckpt["model"])
        raw_discriminator.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        if "scheduler_g" in ckpt:
            scheduler_g.load_state_dict(ckpt["scheduler_g"])
        if "scheduler_d" in ckpt:
            scheduler_d.load_state_dict(ckpt["scheduler_d"])
        elif is_main:
            tqdm.write(
                "NOTE: resuming from a checkpoint saved before LR warmup was added -- scheduler_g/"
                "scheduler_d state not found, starting both schedulers fresh from step 0 of their own "
                "warmup windows. No effect if the checkpoint's step is already past lr_warmup_steps."
            )
        step = ckpt["step"]
        epoch = ckpt["epoch"]
        sampler.set_state(epoch, ckpt["batch_offset"])
        if "rng_state" in ckpt:
            rng = ckpt["rng_state"]
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            if rng["torch_cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in rng["torch_cuda"]])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        elif is_main:
            tqdm.write(
                "NOTE: resuming from a checkpoint saved before RNG state was recorded -- "
                "continuing with a freshly-seeded RNG stream instead of the exact resumed one."
            )
        if is_main:
            print(f"Resumed from {config.train.resume_from} at epoch {epoch}, step {step}")

    model = raw_model
    discriminator = raw_discriminator
    if is_distributed:
        # note: with use_ema=True, DDP re-broadcasts buffers (incl. the EMA cluster
        # stats) from rank 0 before every forward, so only rank 0's local batches
        # actually drive the EMA update -- other ranks' updates get overwritten each
        # step. Not incorrect, just means EMA converges on rank-0-only statistics.
        model = DistributedDataParallel(raw_model, device_ids=[local_rank])
        discriminator = DistributedDataParallel(raw_discriminator, device_ids=[local_rank])

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if is_main else None
    amp_dtype = torch.bfloat16 if config.train.mixed_precision else torch.float32

    def make_loader():
        return DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=config.train.num_workers,
            pin_memory=True,
        )

    model.train()
    discriminator.train()

    if is_main:
        print(
            f"world_size={world_size} batch_size={config.train.batch_size} "
            f"accumulation_steps={config.train.accumulation_steps} "
            f"target_epochs={config.train.target_epochs} "
            f"steps_per_epoch={steps_per_epoch} total_steps={config.train.total_steps}"
        )
    # total_steps is derived from target_epochs * steps_per_epoch and is a fixed target
    # for this run, computed once at startup from the actual dataset -- see Training
    # Length section above. If the dataset or batch_size/accumulation_steps changes
    # later, total_steps is simply recomputed fresh next run (target_epochs itself
    # doesn't need to change to keep seeing the same amount of data per epoch).

    # Dual epoch+step progress display (per .claude/mulit-vqgan.md, Progress Reporting):
    # pbar's own n/total track the persistent *overall* step count across resumes
    # (step/total_steps); the description is refreshed each step with the current
    # epoch and this-epoch step progress, since epoch is the primary training-length
    # unit and shouldn't be buried behind a raw step counter.
    pbar = None
    if is_main:
        pbar = tqdm(
            total=config.train.total_steps,
            initial=step,
            unit="step",
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
        )
    step_durations: deque = deque(maxlen=STEP_TIME_WINDOW)
    effective_step_start = time.perf_counter()
    ddp_revival_warning_shown = False
    last_consumed = 0  # batches consumed so far this epoch; used for the best-effort save on Ctrl+C

    try:
      while step < config.train.total_steps:
        loader = make_loader()
        batch_offset_at_epoch_start = sampler.batch_offset  # accounts for resume offset
        # checkpoints only ever save at an accumulation boundary, so batch_offset is
        # always an exact multiple of accumulation_steps -- safe to convert to a step count
        step_at_epoch_start = batch_offset_at_epoch_start // config.train.accumulation_steps

        for batch_idx, (images, batch_bucket_ids) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            bucket_id = batch_bucket_ids[0].item()
            bucket_name = buckets[bucket_id].name
            is_accum_boundary = (batch_idx + 1) % config.train.accumulation_steps == 0

            next_step = step + 1
            revive_now = (
                config.model.dead_code_revival and is_accum_boundary
                and next_step % config.train.dead_code_revival_every == 0
            )

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=config.train.mixed_precision):
                out = model(images, revive_dead=revive_now)
                x_recon = out["reconstruction"]

                fake_logits = None
                if criterion.discriminator_active(step):
                    fake_logits = discriminator(x_recon)

                gen_losses = criterion.generator_loss(
                    images, x_recon, out["codebook_loss"], step, fake_logits=fake_logits
                )
                gen_loss = gen_losses["total"] / config.train.accumulation_steps

            gen_loss.backward()

            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(generator_params, max_norm=1.0)
                opt_g.step()
                opt_g.zero_grad(set_to_none=True)
                scheduler_g.step()

            disc_loss_value = torch.zeros(())
            if criterion.discriminator_active(step):
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=config.train.mixed_precision):
                    real_logits = discriminator(images)
                    fake_logits_d = discriminator(x_recon.detach())
                    disc_loss = criterion.discriminator_loss_fn(real_logits, fake_logits_d)
                    disc_loss_scaled = disc_loss / config.train.accumulation_steps

                disc_loss_scaled.backward()
                disc_loss_value = disc_loss.detach()

                if is_accum_boundary:
                    torch.nn.utils.clip_grad_norm_(raw_discriminator.parameters(), max_norm=1.0)
                    opt_d.step()
                    opt_d.zero_grad(set_to_none=True)
                    scheduler_d.step()

            if is_accum_boundary:
                step += 1
                last_consumed = batch_offset_at_epoch_start + (batch_idx + 1)
                step_in_epoch = step_at_epoch_start + (batch_idx + 1) // config.train.accumulation_steps

                if is_main:
                    # sync so the measured duration reflects actual GPU-bound compute time
                    # for this full accumulation cycle, not just CPU kernel-launch time
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    now = time.perf_counter()
                    step_duration = now - effective_step_start

                    if step_durations:
                        rolling_avg = sum(step_durations) / len(step_durations)
                        if step_duration > STEP_TIME_SPIKE_FACTOR * rolling_avg:
                            tqdm.write(
                                f"WARNING: step {step} took {step_duration:.2f}s, "
                                f"{step_duration / rolling_avg:.1f}x the rolling average "
                                f"({rolling_avg:.2f}s) -- likely VRAM spillover into shared/system "
                                f"memory rather than a normal slowdown"
                            )
                    step_durations.append(step_duration)
                    rolling_avg = sum(step_durations) / len(step_durations)
                    effective_step_start = now

                    eta_seconds = (config.train.total_steps - step) * rolling_avg
                    pbar.set_description(
                        f"Epoch {epoch + 1}/{config.train.target_epochs} | "
                        f"step {step_in_epoch}/{steps_per_epoch} (this epoch)"
                    )
                    pbar.set_postfix(
                        {
                            "bucket": bucket_name,
                            "s/step": f"{rolling_avg:.2f}",
                            "ETA": format_duration(eta_seconds),
                            "recon": f"{gen_losses['recon_loss']:.4f}",
                            "perc": f"{gen_losses['perceptual_loss']:.4f}",
                            "adv": f"{gen_losses['adversarial_loss']:.4f}",
                        },
                        refresh=False,
                    )
                    pbar.update(1)  # whole steps (weight updates) only -- never fractional/micro-batch counts

                    if device.type == "cuda":
                        writer.add_scalar("system/vram_allocated_gb", torch.cuda.memory_allocated() / 1e9, step)
                        writer.add_scalar("system/vram_reserved_gb", torch.cuda.memory_reserved() / 1e9, step)
                        # note: reflects rank 0's GPU only, not an aggregate across ranks

                    # logged every step (not log_every-gated): cheap scalars, and the
                    # kind of bursty/step-local signal that's most useful at full resolution
                    # while debugging codebook collapse
                    writer.add_scalar("train/codebook_perplexity", out["perplexity"], step)
                    writer.add_scalar("train/codebook_usage", out["codebook_usage"], step)
                    writer.add_scalar("train/codebook_num_revived", out["num_revived"], step)
                    writer.add_scalar("train/epoch", epoch, step)

                    if out["num_revived"] > 0 and is_distributed and not ddp_revival_warning_shown:
                        tqdm.write(
                            "WARNING: dead-code revival just mutated embedding.weight.data directly "
                            "-- this is NOT broadcast to other DDP ranks, so each rank's codebook may "
                            "now differ. Not fixed this round (single-GPU correctness only); a future "
                            "change should dist.broadcast rank 0's revived rows to all ranks."
                        )
                        ddp_revival_warning_shown = True

                    if step % config.train.log_every == 0:
                        writer.add_scalar("train/recon_loss", gen_losses["recon_loss"], step)
                        writer.add_scalar("train/codebook_loss", out["codebook_loss"].detach(), step)
                        writer.add_scalar("train/commitment_loss", out["commitment_loss"], step)
                        writer.add_scalar("train/perceptual_loss", gen_losses["perceptual_loss"], step)
                        writer.add_scalar("train/adversarial_loss", gen_losses["adversarial_loss"], step)
                        writer.add_scalar("train/discriminator_loss", disc_loss_value, step)
                        writer.add_scalar("train/lr_generator", opt_g.param_groups[0]["lr"], step)
                        writer.add_scalar("train/lr_discriminator", opt_d.param_groups[0]["lr"], step)

                        # per-bucket breakdown -- quality can differ across buckets and shouldn't
                        # get averaged away; tagged by the bucket this particular step trained on
                        writer.add_scalar(f"train_bucket_{bucket_name}/recon_loss", gen_losses["recon_loss"], step)
                        writer.add_scalar(f"train_bucket_{bucket_name}/perceptual_loss", gen_losses["perceptual_loss"], step)
                        writer.add_scalar(f"train_bucket_{bucket_name}/adversarial_loss", gen_losses["adversarial_loss"], step)
                        writer.add_scalar(f"train_bucket_{bucket_name}/discriminator_loss", disc_loss_value, step)

                        tqdm.write(
                            f"epoch {epoch + 1}/{config.train.target_epochs} | step {step} | bucket {bucket_name} "
                            f"| recon {gen_losses['recon_loss']:.4f} "
                            f"| perc {gen_losses['perceptual_loss']:.4f} "
                            f"| adv {gen_losses['adversarial_loss']:.4f} "
                            f"| disc {disc_loss_value:.4f} "
                            f"| perplexity {out['perplexity']:.1f} "
                            f"| revived {out['num_revived']}"
                        )

                    if step % config.train.image_log_every == 0:
                        n = min(4, images.size(0))
                        grid_input = denormalize(images[:n])
                        grid_recon = denormalize(x_recon[:n].float())
                        side_by_side = torch.cat([grid_input, grid_recon], dim=3)
                        # tagged per bucket so TensorBoard shows reconstructions across every
                        # aspect ratio as training cycles through buckets, not just one
                        writer.add_images(f"train_bucket_{bucket_name}/input_vs_reconstruction", side_by_side, step)

                    if step % config.train.checkpoint_every == 0:
                        save_checkpoint(
                            ckpt_dir / f"step_{step}.pt", step, epoch, last_consumed, raw_model, raw_discriminator,
                            opt_g, opt_d, scheduler_g, scheduler_d, config,
                        )
                        save_checkpoint(
                            ckpt_dir / "latest.pt", step, epoch, last_consumed, raw_model, raw_discriminator,
                            opt_g, opt_d, scheduler_g, scheduler_d, config,
                        )
                        prune_old_checkpoints(
                            ckpt_dir, config.train.keep_last_n_checkpoints, config.train.checkpoint_milestone_every
                        )
                        tqdm.write(f"Checkpoint saved at epoch {epoch + 1}, step {step}")

                    if val_dataset is not None and step % config.train.val_every == 0:
                        # uses raw_model directly (bypasses the DDP wrapper) -- validation
                        # is forward-only (@torch.no_grad()), so no gradient sync is needed
                        # and this can't desync/stall the other ranks
                        run_validation(raw_model, val_dataset, buckets, device, writer, step, config)
                        raw_model.train()

                if step >= config.train.total_steps:
                    break

        epoch += 1
    except KeyboardInterrupt:
        if is_main:
            tqdm.write(f"\nKeyboardInterrupt at epoch {epoch + 1}, step {step} -- saving a final checkpoint before exit...")
            save_checkpoint(
                ckpt_dir / "latest.pt", step, epoch, last_consumed, raw_model, raw_discriminator,
                opt_g, opt_d, scheduler_g, scheduler_d, config,
            )
            tqdm.write(f"Checkpoint saved at step {step}. Resume with the same command (auto-resumes from latest.pt).")

    if is_main:
        pbar.close()
        writer.close()
    if is_distributed:
        dist.destroy_process_group()


@torch.no_grad()
def run_validation(model: VQGAN, val_dataset, buckets: list, device, writer: SummaryWriter, step: int,
                    config: Config) -> None:
    """Cheap smoke-check subset (overall + per-bucket recon L1). Full eval lives in eval.py."""
    model.eval()
    val_sampler = BucketedBatchSampler(
        val_dataset.bucket_ids, batch_size=config.train.batch_size, seed=config.train.seed, drop_last=False
    )
    loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=2)

    total_l1, total_batches = 0.0, 0
    per_bucket_l1: dict[str, list] = {}
    for images, batch_bucket_ids in loader:
        images = images.to(device)
        out = model(images)
        l1 = F.l1_loss(out["reconstruction"], images).item()
        total_l1 += l1
        total_batches += 1

        bucket_name = buckets[batch_bucket_ids[0].item()].name
        per_bucket_l1.setdefault(bucket_name, []).append(l1)

        if total_batches >= 40:  # cheap subset across buckets, not a full pass
            break

    writer.add_scalar("val/recon_l1", total_l1 / max(total_batches, 1), step)
    for bucket_name, values in per_bucket_l1.items():
        writer.add_scalar(f"val_bucket_{bucket_name}/recon_l1", sum(values) / len(values), step)


if __name__ == "__main__":
    main()
