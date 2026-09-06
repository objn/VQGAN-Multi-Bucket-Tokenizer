"""Stage 2: train the ViT-VQGAN (ViT encoder + quantizer + ViT decoder + CNN discriminator).

Usage:
    python scripts/train_vqgan.py --max-steps 200000
    python scripts/train_vqgan.py --resume checkpoints/vqgan_last.pt
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqgan.config import VQGANTrainConfig
from vqgan.data import CropDataset, build_shards, build_val_batch
from vqgan.display import console, tqdm
from vqgan.models import VQGAN, PatchDiscriminator
from vqgan.training import train_step


def cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float = 0.0) -> float:
    """Cosine decay from base_lr at step 0 down to min_lr at the final step —
    never below min_lr, however low the schedule would otherwise push it."""
    progress = min(step / max(1, total_steps - 1), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


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


def infinite(loader):
    """Cycle a DataLoader forever. Re-entering the loader restarts its workers,
    which is also what reshuffles shard order and crop positions for the next
    pass (see CropDataset.__iter__)."""
    while True:
        yield from loader


def load_index(path):
    index_path = Path(path)
    if not index_path.is_file():
        raise FileNotFoundError(f"{index_path} not found — run scripts/build_index.py first")
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(cfg.index_path)
    train_ds = CropDataset(
        build_shards(index, "train"),
        tile_size=cfg.tile_size,
        crops_per_image=cfg.crops_per_image,
        shuffle_buffer=cfg.shuffle_buffer,
        augment=True,
        seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        drop_last=True, pin_memory=True,
    )
    console.print(f"train shards: {len(train_ds.shards)}")

    console.print("building fixed validation batch...")
    val_images = build_val_batch(
        build_shards(index, "validation"), tile_size=cfg.tile_size, num_images=cfg.val_images
    )
    console.print(f"val crops: {val_images.shape[0]}")
    fixed_val_images = val_images[:8].to(device)

    global_step = 0
    ema_switched = False
    resume_ckpt = None

    # The shape of the network is a fixed property of an already-trained
    # checkpoint — read it from the checkpoint itself on resume rather than
    # from cfg (whose defaults can drift between runs), otherwise
    # load_state_dict below fails with a shape mismatch the moment the two
    # disagree.
    model_config = cfg.model_config()
    if cfg.resume:
        resume_ckpt = torch.load(cfg.resume, map_location=device)
        if "model_config" not in resume_ckpt:
            raise ValueError(
                f"{cfg.resume} has no model_config entry — it predates the ViT-VQGAN rewrite and "
                f"holds CNN encoder/decoder weights that cannot be loaded. Move it aside and "
                f"train from scratch."
            )
        model_config = resume_ckpt["model_config"]
        if model_config != cfg.model_config():
            console.print(
                "[yellow]resume:[/yellow] using the architecture stored in the checkpoint, "
                "not the CLI defaults"
            )

    if model_config["image_size"] != cfg.tile_size:
        raise ValueError(
            f"checkpoint was trained at tile_size {model_config['image_size']} but --tile-size is "
            f"{cfg.tile_size}; the learned position embeddings are tied to the token grid"
        )

    # Start with gradient-based codebook updates; switch to EMA after
    # ema_warmup_steps once the encoder has stabilized (EMA from step 0 can
    # lock in a noisy initial encoder).
    vqgan = VQGAN(**model_config, use_ema=False).to(device)
    discriminator = PatchDiscriminator().to(device)
    grid_size = model_config["image_size"] // model_config["patch_size"]
    n_params = sum(p.numel() for p in vqgan.parameters())
    console.print(f"generator: {n_params / 1e6:.1f}M params, token grid {grid_size}x{grid_size}")

    if resume_ckpt is not None:
        vqgan.load_state_dict(resume_ckpt["vqgan"])
        discriminator.load_state_dict(resume_ckpt["discriminator"])

        # EMA on/off is a plain Python attribute, not part of state_dict, so
        # restore it directly (not via quantizer.set_use_ema(), which would
        # wipe the just-loaded EMA buffers thinking it is switching mode for
        # the first time).
        ema_switched = resume_ckpt.get("ema_switched", False)
        if ema_switched:
            vqgan.quantizer.use_ema = True
            vqgan.quantizer.codebook.weight.requires_grad_(False)

        global_step = resume_ckpt["global_step"]
        console.print(f"resumed from {cfg.resume} at step {global_step} (ema_switched={ema_switched})")

    opt_g = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vqgan.parameters()), lr=cfg.lr, betas=(0.5, 0.9)
    )
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(0.5, 0.9))

    if resume_ckpt is not None:
        if "opt_g" in resume_ckpt:
            opt_g.load_state_dict(resume_ckpt["opt_g"])
        if "opt_d" in resume_ckpt:
            opt_d.load_state_dict(resume_ckpt["opt_d"])

    tb_log_dir = out_dir / "tensorboard"
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
    try:
        subprocess.Popen(["tensorboard", "--logdir", str(tb_log_dir), "--port", "6006"])
        console.print("[cyan]tensorboard:[/cyan] http://localhost:6006")
    except FileNotFoundError:
        console.print(
            f"[yellow]tensorboard CLI not found on PATH[/yellow] — logs are still written to {tb_log_dir}"
        )

    vqgan.train()
    discriminator.train()
    running = {}
    running_n = 0
    pbar = tqdm(initial=global_step, total=cfg.max_steps, desc="train")

    for images in infinite(train_loader):
        if global_step >= cfg.max_steps:
            break

        if not ema_switched and global_step >= cfg.ema_warmup_steps:
            vqgan.quantizer.set_use_ema(True)
            opt_g = torch.optim.Adam(
                filter(lambda p: p.requires_grad, vqgan.parameters()), lr=cfg.lr, betas=(0.5, 0.9)
            )
            ema_switched = True
            console.print(f"step {global_step}: [bold cyan]switched quantizer to EMA mode[/bold cyan]")

        images = images.to(device, non_blocking=True)

        lr = cosine_lr(global_step, cfg.max_steps, cfg.lr, cfg.min_lr)
        for group in opt_g.param_groups:
            group["lr"] = lr
        for group in opt_d.param_groups:
            group["lr"] = lr

        logs = train_step(
            vqgan, discriminator, opt_g, opt_d, images,
            l2_weight=cfg.l2_weight,
            logit_laplace_weight=cfg.logit_laplace_weight,
            lpips_weight=cfg.lpips_weight,
            disc_weight=cfg.disc_weight,
            use_lpips=cfg.use_lpips,
            global_step=global_step,
            disc_start_step=cfg.disc_warmup_steps,
            amp=amp,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        for k, v in logs.items():
            if v is not None:
                running[k] = running.get(k, 0.0) + v
        running_n += 1
        global_step += 1
        pbar.update(1)

        if global_step % cfg.log_every == 0:
            avg = {k: v / running_n for k, v in running.items()}
            postfix = {k: f"{v:.4f}" for k, v in avg.items()}
            postfix["lr"] = f"{lr:.2e}"
            pbar.set_postfix(postfix)
            for k, v in avg.items():
                tb_writer.add_scalar(f"train/{k}", v, global_step)
            tb_writer.add_scalar("train/lr", lr, global_step)
            running, running_n = {}, 0

        if global_step % cfg.eval_every_steps == 0:
            evaluate(vqgan, val_images, device, out_dir, global_step, fixed_val_images,
                     tb_writer, cfg.batch_size)
            vqgan.train()

        if global_step % cfg.checkpoint_every_steps == 0:
            save_checkpoint(vqgan, discriminator, opt_g, opt_d, model_config,
                            global_step, ema_switched, checkpoint_dir)

    pbar.close()
    save_checkpoint(vqgan, discriminator, opt_g, opt_d, model_config,
                    global_step, ema_switched, checkpoint_dir, tag="last")
    tb_writer.close()


@torch.no_grad()
def evaluate(vqgan, val_images, device, out_dir, step, fixed_val_images, tb_writer, batch_size):
    vqgan.eval()
    vqgan.quantizer.reset_usage_stats()

    total_l1, n = 0.0, 0
    for i in range(0, val_images.shape[0], batch_size):
        images = val_images[i:i + batch_size].to(device)
        recon = vqgan(images).recon
        total_l1 += (recon - images).abs().mean().item() * images.shape[0]
        n += images.shape[0]
    val_l1 = total_l1 / max(n, 1)

    usage_pct = vqgan.quantizer.codebook_usage_pct()
    console.print(f"step {step}: val L1 {val_l1:.4f}  codebook usage {usage_pct:.1f}%")
    tb_writer.add_scalar("val/l1", val_l1, step)
    tb_writer.add_scalar("val/codebook_usage_pct", usage_pct, step)

    recon = vqgan(fixed_val_images).recon
    grid = make_grid(torch.cat([fixed_val_images, recon], dim=0), nrow=fixed_val_images.shape[0])
    grid = (grid + 1) / 2
    save_image(grid, out_dir / f"recon_step{step:07d}.png")
    tb_writer.add_image("val/recon", grid, step)


def save_checkpoint(
    vqgan, discriminator, opt_g, opt_d, model_config, global_step, ema_switched,
    checkpoint_dir, tag=None,
):
    name = f"vqgan_{tag}" if tag else f"vqgan_step{global_step:07d}"
    path = checkpoint_dir / f"{name}.pt"
    ckpt = {
        "global_step": global_step,
        "ema_switched": ema_switched,
        "model_config": model_config,
        "vqgan": vqgan.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }
    torch.save(ckpt, path)
    console.print(f"[green]saved[/green] {path}")


if __name__ == "__main__":
    main()
