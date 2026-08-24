"""Stage 2: train the VQGAN (encoder + quantizer + decoder + discriminator).

Usage:
    python scripts/train_vqgan.py --data-dir data/processed --epochs 80
    python scripts/train_vqgan.py --resume checkpoints/vqgan_last.pt
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqgan.config import VQGANTrainConfig
from vqgan.data import PixelDataset
from vqgan.models import VQGAN, PatchDiscriminator
from vqgan.training import train_step


def cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    """Cosine decay from base_lr at step 0 down to ~0 at the final step."""
    progress = min(step / max(1, total_steps - 1), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def parse_args(argv=None) -> VQGANTrainConfig:
    defaults = VQGANTrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for field, default in defaults.__dict__.items():
        flag = "--" + field.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, type=lambda s: s.lower() != "false", default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    args = parser.parse_args(argv)
    return VQGANTrainConfig(**vars(args))


def unwrap(model):
    """The underlying module, whether or not it's DDP-wrapped — use this for
    checkpoint save/load and for reaching through to .quantizer etc., so
    checkpoints stay identical regardless of how many GPUs produced them."""
    return model.module if isinstance(model, DDP) else model


def setup_distributed(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    # NCCL isn't available on Windows builds of PyTorch; gloo covers both
    # CPU and CUDA tensors there.
    backend = "nccl" if dist.is_nccl_available() and sys.platform != "win32" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank: int = 0, world_size: int = 1, argv=None):
    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)

    distributed = world_size > 1
    if distributed:
        setup_distributed(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    amp = cfg.amp and device.type == "cuda"

    checkpoint_dir = Path(cfg.checkpoint_dir)
    out_dir = Path(cfg.out_dir)
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()  # make every rank wait until rank 0 has made the dirs

    train_ds = PixelDataset(cfg.data_dir, split="train")
    val_ds = PixelDataset(cfg.data_dir, split="val")

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed)
        if distributed else None
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=cfg.num_workers, drop_last=False, pin_memory=True,
    )
    # Left un-sharded: only rank 0 ever runs evaluate() (see below), so every
    # rank building the full val_loader (rather than plumbing a distributed
    # one through unused on non-main ranks) keeps this simpler.
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    if is_main:
        suffix = f"  (world_size={world_size})" if distributed else ""
        print(f"train: {len(train_ds)}  val: {len(val_ds)}{suffix}")

    start_epoch = 0
    global_step = 0
    ema_switched = False
    resume_ckpt = None

    # latent_dim/num_embeddings are fixed properties of an already-trained
    # checkpoint — read them from the checkpoint itself on resume rather than
    # cfg (whose defaults can drift between runs), otherwise load_state_dict
    # below fails with a codebook shape mismatch the moment cfg's defaults
    # disagree with what this checkpoint was actually trained with.
    latent_dim, num_embeddings = cfg.latent_dim, cfg.num_embeddings
    if cfg.resume:
        resume_ckpt = torch.load(cfg.resume, map_location=device)
        latent_dim = resume_ckpt["latent_dim"]
        num_embeddings = resume_ckpt["num_embeddings"]
        if is_main and (latent_dim != cfg.latent_dim or num_embeddings != cfg.num_embeddings):
            print(
                f"resume: overriding --latent-dim/--num-embeddings with checkpoint's "
                f"own values ({latent_dim}, {num_embeddings})"
            )

    # Start with gradient-based codebook updates; switch to EMA after
    # ema_warmup_epochs once the encoder has stabilized (spec's recommended
    # workflow — EMA from step 0 can lock in a noisy initial encoder).
    vqgan = VQGAN(latent_dim=latent_dim, num_embeddings=num_embeddings, use_ema=False).to(device)
    discriminator = PatchDiscriminator().to(device)

    if resume_ckpt is not None:
        vqgan.load_state_dict(resume_ckpt["vqgan"])
        discriminator.load_state_dict(resume_ckpt["discriminator"])

        # EMA on/off is a plain Python attribute, not part of state_dict, so
        # restore it directly (not via quantizer.set_use_ema(), which would
        # wipe the just-loaded EMA buffers thinking it's switching mode for
        # the first time).
        ema_switched = resume_ckpt.get("ema_switched", False)
        if ema_switched:
            vqgan.quantizer.use_ema = True
            vqgan.quantizer.codebook.weight.requires_grad_(False)

        start_epoch = resume_ckpt["epoch"] + 1
        global_step = resume_ckpt.get("global_step", start_epoch * len(train_loader))
        if is_main:
            print(f"resumed from {cfg.resume} at epoch {start_epoch} (ema_switched={ema_switched})")

    if distributed:
        # find_unused_parameters=True: the quantizer's codebook stops
        # requiring grad partway through training (the EMA warmup switch
        # below), which changes which parameters show up in the autograd
        # graph between iterations — DDP needs to re-check each forward
        # rather than assume a fixed parameter set.
        vqgan = DDP(vqgan, device_ids=[rank], find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[rank])

    opt_g = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vqgan.parameters()), lr=cfg.lr, betas=(0.5, 0.9)
    )
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(0.5, 0.9))

    if resume_ckpt is not None:
        if "opt_g" in resume_ckpt:
            opt_g.load_state_dict(resume_ckpt["opt_g"])
        if "opt_d" in resume_ckpt:
            opt_d.load_state_dict(resume_ckpt["opt_d"])

    steps_per_epoch = len(train_loader)
    disc_start_step = int(cfg.disc_warmup_epochs * steps_per_epoch)
    total_steps = cfg.epochs * steps_per_epoch

    fixed_val_batch = next(iter(val_loader))
    fixed_val_images = fixed_val_batch[0][:8].to(device)

    tb_writer = None
    if is_main:
        tb_log_dir = out_dir / "tensorboard"
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        try:
            subprocess.Popen(["tensorboard", "--logdir", str(tb_log_dir), "--port", "6006"])
            print("tensorboard: http://localhost:6006")
        except FileNotFoundError:
            print(f"tensorboard CLI not found on PATH — logs are still written to {tb_log_dir}")

    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if not ema_switched and epoch >= cfg.ema_warmup_epochs:
            unwrap(vqgan).quantizer.set_use_ema(True)
            opt_g = torch.optim.Adam(
                filter(lambda p: p.requires_grad, vqgan.parameters()), lr=cfg.lr, betas=(0.5, 0.9)
            )
            ema_switched = True
            if is_main:
                print(f"epoch {epoch}: switched quantizer to EMA mode")

        vqgan.train()
        discriminator.train()
        running = {}
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_main)
        for step, (images, valid_mask, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

            lr = cosine_lr(global_step, total_steps, cfg.lr)
            for g in opt_g.param_groups:
                g["lr"] = lr
            for g in opt_d.param_groups:
                g["lr"] = lr

            logs = train_step(
                vqgan, discriminator, opt_g, opt_d, images,
                valid_mask=valid_mask,
                disc_weight=cfg.disc_weight,
                use_lpips=cfg.use_lpips,
                lpips_weight=cfg.lpips_weight,
                global_step=global_step,
                disc_start_step=disc_start_step,
                amp=amp,
                grad_clip_norm=cfg.grad_clip_norm,
            )
            for k, v in logs.items():
                if v is not None:
                    running[k] = running.get(k, 0.0) + v
            global_step += 1

            if is_main and step % cfg.log_every == 0:
                avg = {k: v / (step + 1) for k, v in running.items()}
                postfix = {k: f"{v:.4f}" for k, v in avg.items()}
                postfix["lr"] = f"{lr:.2e}"
                pbar.set_postfix(postfix)

        usage_count = unwrap(vqgan).quantizer.usage_count
        if distributed:
            # Each rank only scattered its own shard of the batch into
            # usage_count — sum across ranks so the reported percentage
            # reflects codebook usage over the whole global batch.
            dist.all_reduce(usage_count, op=dist.ReduceOp.SUM)
        usage_pct = unwrap(vqgan).quantizer.codebook_usage_pct()
        unwrap(vqgan).quantizer.reset_usage_stats()
        if is_main:
            print(f"epoch {epoch}: codebook usage {usage_pct:.1f}%")

        if is_main:
            epoch_avg = {k: v / steps_per_epoch for k, v in running.items()}
            for k, v in epoch_avg.items():
                tb_writer.add_scalar(f"train/{k}", v, epoch)
            tb_writer.add_scalar("train/codebook_usage_pct", usage_pct, epoch)
            tb_writer.add_scalar("train/lr", lr, epoch)

        if is_main and epoch % cfg.eval_every_epochs == 0:
            # unwrap(vqgan): DDP's forward() does a collective buffer-sync
            # that every rank must join, but only rank 0 runs eval — call
            # the plain underlying module instead (its weights are already
            # kept in sync by DDP's gradient all-reduce during training) to
            # avoid the other ranks hanging waiting for a broadcast that
            # never comes.
            evaluate(unwrap(vqgan), val_loader, device, out_dir, epoch, fixed_val_images, tb_writer)

        if is_main and (epoch % cfg.checkpoint_every_epochs == 0 or epoch == cfg.epochs - 1):
            save_checkpoint(
                unwrap(vqgan), unwrap(discriminator), opt_g, opt_d,
                latent_dim, num_embeddings, epoch, global_step, ema_switched, checkpoint_dir,
            )

        if distributed:
            dist.barrier()

    if is_main:
        save_checkpoint(
            unwrap(vqgan), unwrap(discriminator), opt_g, opt_d,
            latent_dim, num_embeddings, cfg.epochs - 1, global_step, ema_switched,
            checkpoint_dir, tag="last",
        )
        tb_writer.close()

    if distributed:
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(vqgan, val_loader, device, out_dir, epoch, fixed_val_images, tb_writer):
    vqgan.eval()
    total_l1, n = 0.0, 0
    for images, valid_mask, _ in val_loader:
        images = images.to(device)
        recon, _, _ = vqgan(images)
        total_l1 += (recon - images).abs().mean().item() * images.shape[0]
        n += images.shape[0]
    val_l1 = total_l1 / max(n, 1)
    print(f"epoch {epoch}: val L1 {val_l1:.4f}")
    tb_writer.add_scalar("val/l1", val_l1, epoch)

    recon, _, _ = vqgan(fixed_val_images)
    grid = make_grid(torch.cat([fixed_val_images, recon], dim=0), nrow=fixed_val_images.shape[0])
    grid = (grid + 1) / 2
    save_image(grid, out_dir / f"recon_epoch{epoch:04d}.png")
    tb_writer.add_image("val/recon", grid, epoch)


def save_checkpoint(
    vqgan, discriminator, opt_g, opt_d,
    latent_dim, num_embeddings, epoch, global_step, ema_switched, checkpoint_dir, tag=None,
):
    name = f"vqgan_{tag}" if tag else f"vqgan_epoch{epoch:04d}"
    path = checkpoint_dir / f"{name}.pt"
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "ema_switched": ema_switched,
        "vqgan": vqgan.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "latent_dim": latent_dim,
        "num_embeddings": num_embeddings,
    }
    torch.save(ckpt, path)
    print(f"saved {path}")


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"detected {world_size} GPUs — launching DDP training")
        mp.spawn(main, args=(world_size, sys.argv[1:]), nprocs=world_size, join=True)
    else:
        main(0, 1, sys.argv[1:])
