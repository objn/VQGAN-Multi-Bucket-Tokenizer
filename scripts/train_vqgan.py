"""Stage 2: train the VQGAN (encoder + quantizer + decoder + discriminator).

Usage:
    python scripts/train_vqgan.py --data-dir data/processed --epochs 80
    python scripts/train_vqgan.py --resume checkpoints/vqgan_last.pt
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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


def main(argv=None):
    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PixelDataset(cfg.data_dir, split="train")
    val_ds = PixelDataset(cfg.data_dir, split="val")
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=False, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

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
        if latent_dim != cfg.latent_dim or num_embeddings != cfg.num_embeddings:
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
        print(f"resumed from {cfg.resume} at epoch {start_epoch} (ema_switched={ema_switched})")

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

    tb_log_dir = out_dir / "tensorboard"
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
    try:
        subprocess.Popen(["tensorboard", "--logdir", str(tb_log_dir), "--port", "6006"])
        print("tensorboard: http://localhost:6006")
    except FileNotFoundError:
        print(f"tensorboard CLI not found on PATH — logs are still written to {tb_log_dir}")

    for epoch in range(start_epoch, cfg.epochs):
        if not ema_switched and epoch >= cfg.ema_warmup_epochs:
            vqgan.quantizer.set_use_ema(True)
            opt_g = torch.optim.Adam(
                filter(lambda p: p.requires_grad, vqgan.parameters()), lr=cfg.lr, betas=(0.5, 0.9)
            )
            ema_switched = True
            print(f"epoch {epoch}: switched quantizer to EMA mode")

        vqgan.train()
        discriminator.train()
        running = {}
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
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

            if step % cfg.log_every == 0:
                avg = {k: v / (step + 1) for k, v in running.items()}
                postfix = {k: f"{v:.4f}" for k, v in avg.items()}
                postfix["lr"] = f"{lr:.2e}"
                pbar.set_postfix(postfix)

        usage_pct = vqgan.quantizer.codebook_usage_pct()
        print(f"epoch {epoch}: codebook usage {usage_pct:.1f}%")
        vqgan.quantizer.reset_usage_stats()

        epoch_avg = {k: v / steps_per_epoch for k, v in running.items()}
        for k, v in epoch_avg.items():
            tb_writer.add_scalar(f"train/{k}", v, epoch)
        tb_writer.add_scalar("train/codebook_usage_pct", usage_pct, epoch)
        tb_writer.add_scalar("train/lr", lr, epoch)

        if epoch % cfg.eval_every_epochs == 0:
            evaluate(vqgan, val_loader, device, out_dir, epoch, fixed_val_images, tb_writer)

        if epoch % cfg.checkpoint_every_epochs == 0 or epoch == cfg.epochs - 1:
            save_checkpoint(
                vqgan, discriminator, opt_g, opt_d,
                latent_dim, num_embeddings, epoch, global_step, ema_switched, checkpoint_dir,
            )

    save_checkpoint(
        vqgan, discriminator, opt_g, opt_d,
        latent_dim, num_embeddings, cfg.epochs - 1, global_step, ema_switched,
        checkpoint_dir, tag="last",
    )
    tb_writer.close()


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
    main()
