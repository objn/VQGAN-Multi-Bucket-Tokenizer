"""Main training entrypoint for the VQGAN tokenizer.

Single-GPU usage:
    uv run python -m vqgan.train --train-dir data/train --val-dir data/val

Multi-GPU (DistributedDataParallel) usage, one process per GPU via torchrun:
    torchrun --standalone --nproc_per_node=2 -m vqgan.train --train-dir data/train --val-dir data/val
"""
import argparse
import dataclasses
import os
import time
from collections import deque
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vqgan.config import Config, DEFAULT_CONFIG_PATH, load_config
from vqgan.data.buckets import Bucket, BucketedBatchSampler
from vqgan.data.dataset import BucketedImageDataset
from vqgan.losses.losses import VQGANLoss
from vqgan.models.discriminator import PatchGANDiscriminator
from vqgan.models.vqgan import VQGAN

# rolling window size for step-speed/ETA smoothing -- large enough that the first few
# steps' CUDA warmup / cuDNN autotune overhead doesn't dominate the estimate
STEP_TIME_WINDOW = 50
STEP_TIME_SPIKE_FACTOR = 5.0  # console warning threshold: current step vs rolling avg


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


def parse_args() -> Config:
    """All hyperparameters (including the bucket table) come from a single JSON file
    (--config, default configs/vqgan-multi.json). CLI flags below are optional
    overrides on top of it -- only flags actually passed on the command line take
    effect (default=None)."""
    parser = argparse.ArgumentParser(description="Train the VQGAN multi-resolution bucket tokenizer")

    parser.add_argument("--config", type=str, default=None, help=f"path to a JSON config (default: {DEFAULT_CONFIG_PATH})")

    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--lr-generator", type=float, default=None)
    parser.add_argument("--lr-discriminator", type=float, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true", default=None)
    parser.add_argument("--no-grad-checkpointing", action="store_true", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--use-ema", action="store_true", default=None)
    parser.add_argument("--discriminator-start-step", type=int, default=None)
    parser.add_argument("--lambda-perceptual", type=float, default=None)
    parser.add_argument("--lambda-adv", type=float, default=None)

    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--image-log-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.train_dir is not None:
        config.data.train_dir = args.train_dir
    if args.val_dir is not None:
        config.data.val_dir = args.val_dir
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
        config.train.num_workers = args.num_workers

    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    if args.total_steps is not None:
        config.train.total_steps = args.total_steps
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        config.train.grad_accum_steps = args.grad_accum_steps
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

    return config


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for TensorBoard image logging."""
    return (x.clamp(-1, 1) + 1) / 2


def save_checkpoint(path: Path, step: int, epoch: int, batch_offset: int, raw_model, raw_discriminator,
                     opt_g, opt_d, config: Config) -> None:
    """Takes the raw (un-DDP-wrapped) model/discriminator so checkpoints have clean
    state_dict keys (no "module." prefix) and load identically in single-GPU or
    multi-GPU runs, and in eval.py (which always builds a plain VQGAN)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "batch_offset": batch_offset,
            "model": raw_model.state_dict(),
            "discriminator": raw_discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "config": dataclasses.asdict(config),
        },
        path,
    )


def main() -> None:
    config = parse_args()
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

    train_dataset = BucketedImageDataset(
        config.data.train_dir, buckets=buckets, horizontal_flip=config.data.horizontal_flip, train=True
    )
    val_dataset = None
    if Path(config.data.val_dir).exists():
        val_dataset = BucketedImageDataset(config.data.val_dir, buckets=buckets, train=False)

    sampler = BucketedBatchSampler(
        train_dataset.bucket_ids, batch_size=config.train.batch_size, seed=config.train.seed,
        rank=rank, world_size=world_size,
    )

    step = 0
    epoch = 0

    if config.train.resume_from:
        ckpt = torch.load(config.train.resume_from, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        raw_discriminator.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        step = ckpt["step"]
        epoch = ckpt["epoch"]
        sampler.set_state(epoch, ckpt["batch_offset"])
        if is_main:
            print(f"Resumed from {config.train.resume_from} at step {step}")

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
        effective_batch_size = config.train.batch_size * config.train.grad_accum_steps * world_size
        print(
            f"world_size={world_size} "
            f"physical_batch_size_per_gpu={config.train.batch_size} "
            f"accumulation_steps={config.train.grad_accum_steps} "
            f"effective_batch_size={effective_batch_size} "
            f"total_effective_steps={config.train.total_steps}"
        )
    # total_effective_steps represents a fixed amount of data the model should see, not
    # a fixed number of physical batches -- it must NOT be recomputed if grad_accum_steps
    # or world_size changes later (e.g. to fit VRAM after lowering physical batch size)

    # bar_format is overridden so the bar only ever shows effective steps (never
    # fractional/physical sub-steps) -- speed and ETA are computed manually below with a
    # rolling average and injected via postfix instead of tqdm's built-in (exponential,
    # not windowed) rate estimate
    pbar = None
    if is_main:
        pbar = tqdm(
            total=config.train.total_steps,
            initial=step,
            unit="step",
            desc="train",
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
        )
    step_durations: deque = deque(maxlen=STEP_TIME_WINDOW)
    effective_step_start = time.perf_counter()

    while step < config.train.total_steps:
        loader = make_loader()
        batch_offset_at_epoch_start = sampler.batch_offset  # accounts for resume offset

        for batch_idx, (images, batch_bucket_ids) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            bucket_name = buckets[batch_bucket_ids[0].item()].name
            is_accum_boundary = (batch_idx + 1) % config.train.grad_accum_steps == 0

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=config.train.mixed_precision):
                out = model(images)
                x_recon = out["reconstruction"]

                fake_logits = None
                if criterion.discriminator_active(step):
                    fake_logits = discriminator(x_recon)

                gen_losses = criterion.generator_loss(
                    images, x_recon, out["codebook_loss"], step, fake_logits=fake_logits
                )
                gen_loss = gen_losses["total"] / config.train.grad_accum_steps

            gen_loss.backward()

            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(generator_params, max_norm=1.0)
                opt_g.step()
                opt_g.zero_grad(set_to_none=True)

            disc_loss_value = torch.zeros(())
            if criterion.discriminator_active(step):
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=config.train.mixed_precision):
                    real_logits = discriminator(images)
                    fake_logits_d = discriminator(x_recon.detach())
                    disc_loss = criterion.discriminator_loss_fn(real_logits, fake_logits_d)
                    disc_loss_scaled = disc_loss / config.train.grad_accum_steps

                disc_loss_scaled.backward()
                disc_loss_value = disc_loss.detach()

                if is_accum_boundary:
                    torch.nn.utils.clip_grad_norm_(raw_discriminator.parameters(), max_norm=1.0)
                    opt_d.step()
                    opt_d.zero_grad(set_to_none=True)

            if is_accum_boundary:
                step += 1

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
                    pbar.update(1)  # effective steps only -- never fractional/physical sub-steps

                    if device.type == "cuda":
                        writer.add_scalar("system/vram_allocated_gb", torch.cuda.memory_allocated() / 1e9, step)
                        writer.add_scalar("system/vram_reserved_gb", torch.cuda.memory_reserved() / 1e9, step)
                        # note: reflects rank 0's GPU only, not an aggregate across ranks

                    if step % config.train.log_every == 0:
                        writer.add_scalar("train/recon_loss", gen_losses["recon_loss"], step)
                        writer.add_scalar("train/codebook_loss", out["codebook_loss"].detach(), step)
                        writer.add_scalar("train/commitment_loss", out["commitment_loss"], step)
                        writer.add_scalar("train/perceptual_loss", gen_losses["perceptual_loss"], step)
                        writer.add_scalar("train/adversarial_loss", gen_losses["adversarial_loss"], step)
                        writer.add_scalar("train/discriminator_loss", disc_loss_value, step)
                        writer.add_scalar("train/codebook_perplexity", out["perplexity"], step)
                        writer.add_scalar("train/codebook_usage", out["codebook_usage"], step)
                        writer.add_scalar("train/lr_generator", opt_g.param_groups[0]["lr"], step)
                        writer.add_scalar("train/lr_discriminator", opt_d.param_groups[0]["lr"], step)

                        # per-bucket breakdown -- quality can differ across buckets and shouldn't
                        # get averaged away; tagged by the bucket this particular step trained on
                        writer.add_scalar(f"train_bucket_{bucket_name}/recon_loss", gen_losses["recon_loss"], step)
                        writer.add_scalar(f"train_bucket_{bucket_name}/perceptual_loss", gen_losses["perceptual_loss"], step)
                        writer.add_scalar(f"train_bucket_{bucket_name}/adversarial_loss", gen_losses["adversarial_loss"], step)
                        writer.add_scalar(f"train_bucket_{bucket_name}/discriminator_loss", disc_loss_value, step)

                        tqdm.write(
                            f"step {step} | bucket {bucket_name} | recon {gen_losses['recon_loss']:.4f} "
                            f"| perc {gen_losses['perceptual_loss']:.4f} "
                            f"| adv {gen_losses['adversarial_loss']:.4f} "
                            f"| disc {disc_loss_value:.4f} "
                            f"| perplexity {out['perplexity']:.1f}"
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
                        consumed = batch_offset_at_epoch_start + (batch_idx + 1)
                        save_checkpoint(
                            ckpt_dir / f"step_{step}.pt", step, epoch, consumed, raw_model, raw_discriminator,
                            opt_g, opt_d, config,
                        )
                        save_checkpoint(
                            ckpt_dir / "latest.pt", step, epoch, consumed, raw_model, raw_discriminator,
                            opt_g, opt_d, config,
                        )

                    if val_dataset is not None and step % config.train.val_every == 0:
                        # uses raw_model directly (bypasses the DDP wrapper) -- validation
                        # is forward-only (@torch.no_grad()), so no gradient sync is needed
                        # and this can't desync/stall the other ranks
                        run_validation(raw_model, val_dataset, buckets, device, writer, step, config)
                        raw_model.train()

                if step >= config.train.total_steps:
                    break

        epoch += 1

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
