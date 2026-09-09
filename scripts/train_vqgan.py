"""Stage 2: train the ViT-VQGAN (ViT encoder + quantizer + ViT decoder + CNN discriminator).

The schedule (max_steps, warmups, eval/checkpoint cadence) is counted in
images, not in optimizer steps at whatever batch_size this run uses — see
VQGANTrainConfig's "Schedule" section. batch_size is therefore a pure
throughput knob: warmups, evaluation, checkpoints and the LR decay all land
at the same point in the data whatever it is set to, and a bigger batch just
gets there faster.

Usage:
    python scripts/train_vqgan.py --max-steps 1600000
    python scripts/train_vqgan.py --batch-size 128         # schedule unchanged, just faster
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

from vqgan.config import DataConfig, VQGANTrainConfig
from vqgan.data import CropDataset, build_shards
from vqgan.data.eval_subset import (
    build_balanced_val_subset,
    compute_log_size_edges,
    materialize_selected_crops,
)
from vqgan.data.sources import FolderShard, ParquetShard, count_parquet_rows
from vqgan.display import console, tqdm
from vqgan.models import VQGAN, PatchDiscriminator
from vqgan.training import train_step


def cosine_lr(
    images_seen: int, end_images: int, base_lr: float, min_lr: float = 0.0,
    warmup_images: int = 0,
) -> float:
    """Linear warmup from 0 to base_lr over the first `warmup_images`, then
    cosine decay from base_lr down to min_lr over the images remaining until
    `end_images` — never below min_lr, however low the schedule would
    otherwise push it, and flat at min_lr for any images_seen past
    `end_images` (`end_images` is where the decay ends, not necessarily
    where training ends — see VQGANTrainConfig.end_steps_lr).

    Progress is measured in images rather than steps so the curve a run
    follows is the same curve at any batch size. warmup_images=0 (the
    default) skips straight to the cosine decay at base_lr on image 0.
    """
    if warmup_images > 0 and images_seen < warmup_images:
        return base_lr * images_seen / warmup_images
    progress = min(
        (images_seen - warmup_images) / max(1, end_images - warmup_images - 1), 1.0
    )
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def crossed(previous: int, current: int, every: int) -> bool:
    """Did the count pass a multiple of `every` between these two readings?

    A batch rarely lands exactly on a multiple, so `count % every == 0` would
    skip whole triggers at larger batch sizes (at batch 128, only one image
    count in 128 is even a candidate). Comparing floor-divisions fires once per
    interval crossed no matter how the images were grouped — the same test the
    quantizer uses for dead-code revival.
    """
    return every > 0 and current // every > previous // every


def parse_args(argv=None):
    """-> (config, reset_discriminator). Flags are derived from the dataclass
    fields, so adding a config knob adds its flag automatically; anything that
    is not a property of the run itself (like --reset-discriminator, which is
    about one resume) is added by hand and kept out of the config."""
    defaults = VQGANTrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for field, default in defaults.__dict__.items():
        flag = "--" + field.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, type=lambda s: s.lower() != "false", default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    parser.add_argument(
        "--reset-discriminator", action="store_true",
        help="resume the generator but start the discriminator from scratch",
    )
    args = vars(parser.parse_args(argv))
    reset_discriminator = args.pop("reset_discriminator")
    return VQGANTrainConfig(**args), reset_discriminator


def infinite(loader):
    """Cycle a DataLoader forever. Re-entering the loader restarts its workers,
    which is also what reshuffles shard order and crop positions for the next
    pass (see CropDataset.__iter__)."""
    while True:
        yield from loader


def describe_shards(shards) -> str:
    """Shard counts alone read as dataset size and mislead — 294 parquet files
    hold 1.28M images. Row counts come from parquet footers, so this is free."""
    parquet = [s.path for s in shards if isinstance(s, ParquetShard)]
    folder = sum(len(s.paths) for s in shards if isinstance(s, FolderShard))
    parts = []
    if parquet:
        parts.append(f"{len(parquet)} parquet shard(s) / {count_parquet_rows(parquet):,} images")
    if folder:
        parts.append(f"{folder:,} folder image(s)")
    return ", ".join(parts)


def load_index(path):
    index_path = Path(path)
    if not index_path.is_file():
        raise FileNotFoundError(f"{index_path} not found — run scripts/build_index.py first")
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    cfg, reset_discriminator = parse_args(argv)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(cfg.index_path)
    # The buffer is what stops a batch from being one photo's tile grid, so how
    # much mixing it buys has to be measured against the batch it feeds, not in
    # crops alone.
    shuffle_buffer = max(cfg.shuffle_buffer, 8 * cfg.batch_size)
    # The in-training readout scales with the machine: a card that fits a big
    # batch can afford a longer, less noisy val prefix. Unlike everything else
    # here that makes the val curve comparable only against runs at the same
    # batch size — see VQGANTrainConfig.eval_images.
    eval_images = max(cfg.eval_images, 64 * cfg.batch_size)
    train_ds = CropDataset(
        build_shards(index, "train", cfg.source),
        tile_size=cfg.tile_size,
        tile_overlap_ratio=cfg.tile_overlap_ratio,
        shuffle_buffer=shuffle_buffer,
        augment=True,
        seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        drop_last=True, pin_memory=True,
    )

    # Same crop size and the same grid, minus the per-pass jitter that rides
    # along with `shuffle` — so the nth validation crop is the same crop at
    # every evaluation point.
    val_ds = CropDataset(
        build_shards(index, "validation", cfg.source),
        tile_size=cfg.tile_size,
        tile_overlap_ratio=cfg.tile_overlap_ratio,
        shuffle=False,
        augment=False,
    )
    if not train_ds.shards:
        raise RuntimeError(f"no train shards for source={cfg.source!r} — check {cfg.index_path}")
    if not val_ds.shards:
        raise RuntimeError(f"no validation shards for source={cfg.source!r} — check {cfg.index_path}")

    data_cfg = DataConfig()
    where = {"parquet": data_cfg.manifest_dir, "folder": data_cfg.folder_root,
             "all": f"{data_cfg.manifest_dir} + {data_cfg.folder_root}"}[cfg.source]
    console.print(f"[bold]source[/bold] {cfg.source} ({where}/)")
    for name, shards in (("train", train_ds.shards), ("validation", val_ds.shards)):
        console.print(f"  {name:>10}: {describe_shards(shards)}")

    # A fixed, size-balanced subset of the validation split for the eval
    # readout below — computed once here (header reads only), then decoded
    # once, so every evaluate() call for the rest of the run reuses the same
    # crops instead of re-scanning the split. See VQGANTrainConfig.eval_images
    # and eval_size_groups, and vqgan/data/eval_subset.py.
    size_edges = compute_log_size_edges(
        val_ds.shards, tile_size=cfg.tile_size, num_groups=cfg.eval_size_groups
    )
    eval_selection = build_balanced_val_subset(
        val_ds.shards, tile_size=cfg.tile_size, tile_overlap_ratio=cfg.tile_overlap_ratio,
        edges=size_edges, max_crops_target=eval_images, seed=cfg.seed,
    )
    eval_val_crops, eval_offsets = materialize_selected_crops(
        val_ds.shards, eval_selection, tile_size=cfg.tile_size,
        tile_overlap_ratio=cfg.tile_overlap_ratio,
    )
    floor_note = f" (above the {cfg.eval_images:,} floor, 64 x batch {cfg.batch_size})" \
        if eval_images != cfg.eval_images else ""
    console.print(
        f"[bold]val readout[/bold] {eval_val_crops.shape[0]:,} size-balanced crops from "
        f"{len(eval_selection):,} image(s) across {cfg.eval_size_groups} size group(s), "
        f"seed {cfg.seed}{floor_note}"
    )

    # One representative per size group for the recon-preview grid — the
    # first image the round-robin picked from each group, reusing its
    # already-decoded crops (all of them, not just one) rather than a
    # separate pick.
    seen_groups, preview_slices = set(), []
    for sel, (start, end) in zip(eval_selection, eval_offsets):
        if sel.group_index not in seen_groups:
            seen_groups.add(sel.group_index)
            preview_slices.append((start, end))
    preview_images = torch.cat(
        [eval_val_crops[start:end] for start, end in preview_slices], dim=0
    ).to(device)

    global_step = 0
    images_seen = 0
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
    # ema_warmup_steps once the encoder has stabilized (EMA from image 0 can
    # lock in a noisy initial encoder).
    vqgan = VQGAN(**model_config, use_ema=False).to(device)
    discriminator = PatchDiscriminator().to(device)
    grid_size = model_config["image_size"] // model_config["patch_size"]
    n_params = sum(p.numel() for p in vqgan.parameters())
    console.print(f"generator: {n_params / 1e6:.1f}M params, token grid {grid_size}x{grid_size}")

    if resume_ckpt is not None:
        vqgan.load_state_dict(resume_ckpt["vqgan"])
        if reset_discriminator:
            console.print("[yellow]resume:[/yellow] discriminator reinitialized, generator kept")
        else:
            try:
                discriminator.load_state_dict(resume_ckpt["discriminator"])
            except RuntimeError as e:
                raise ValueError(
                    f"{cfg.resume} holds a discriminator this code cannot load: {e}\n"
                    f"Checkpoints written before the discriminator moved from BatchNorm to "
                    f"GroupNorm carry running_mean/running_var buffers that no longer exist. "
                    f"Pass --reset-discriminator to resume the generator and train a fresh "
                    f"discriminator (disc_warmup_steps gives it room to catch up)."
                ) from e

        # EMA on/off is a plain Python attribute, not part of state_dict, so
        # restore it directly (not via quantizer.set_use_ema(), which would
        # wipe the just-loaded EMA buffers thinking it is switching mode for
        # the first time).
        ema_switched = resume_ckpt.get("ema_switched", False)
        if ema_switched:
            vqgan.quantizer.use_ema = True
            vqgan.quantizer.codebook.weight.requires_grad_(False)

        global_step = resume_ckpt["global_step"]
        # Checkpoints from before the schedule was counted in images only know
        # their step count, which meant reference_batch_size images each.
        images_seen = resume_ckpt.get(
            "images_seen", global_step * cfg.reference_batch_size
        )
        console.print(
            f"resumed from {cfg.resume} at {images_seen:,} images / step {global_step} "
            f"(ema_switched={ema_switched})"
        )

    base_lr = cfg.scaled_lr()
    if base_lr != cfg.lr:
        console.print(
            f"[bold]lr[/bold] {base_lr:.2e} — {cfg.lr:.2e} scaled by {cfg.lr_scaling} "
            f"for batch {cfg.batch_size} vs reference {cfg.reference_batch_size}"
        )
    if cfg.end_steps_lr != cfg.max_steps:
        where = "before" if cfg.end_steps_lr < cfg.max_steps else "past"
        console.print(
            f"[bold]lr decay[/bold] reaches min_lr at {cfg.end_steps_lr:,} images, "
            f"{where} max_steps ({cfg.max_steps:,}) — "
            + ("lr sits flat at min_lr for the rest of the run"
               if cfg.end_steps_lr < cfg.max_steps
               else "lr never reaches min_lr within this run")
        )

    # betas are a momentum window measured in steps, so unlike everything else
    # here they do shift with batch size. Left fixed: (0.5, 0.9) is the pairing
    # GAN training is known to be stable at, and compounding them per image the
    # way the codebook EMA does would leave essentially no momentum at all at
    # large batches (0.5 ** 16 is 1.5e-5).
    opt_g = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vqgan.parameters()), lr=base_lr, betas=(0.5, 0.9)
    )
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=base_lr, betas=(0.5, 0.9))

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
    # Progress, logging and every trigger below are counted in images. Steps
    # are still counted, but only to name checkpoints and to average the
    # running loss over the batches that produced it.
    pbar = tqdm(initial=images_seen, total=cfg.max_steps, desc="train", unit="img")

    for images in infinite(train_loader):
        if images_seen >= cfg.max_steps:
            break

        if not ema_switched and images_seen >= cfg.ema_warmup_steps:
            vqgan.quantizer.set_use_ema(True)
            opt_g = torch.optim.Adam(
                filter(lambda p: p.requires_grad, vqgan.parameters()), lr=base_lr, betas=(0.5, 0.9)
            )
            ema_switched = True
            console.print(
                f"{images_seen:,} images: [bold cyan]switched quantizer to EMA mode[/bold cyan]"
            )

        images = images.to(device, non_blocking=True)

        lr = cosine_lr(images_seen, cfg.end_steps_lr, base_lr, cfg.min_lr, cfg.lr_warmup_steps)
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
            images_seen=images_seen,
            disc_start_images=cfg.disc_warmup_steps,
            amp=amp,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        for k, v in logs.items():
            if v is not None:
                running[k] = running.get(k, 0.0) + v
        running_n += 1
        global_step += 1
        previous_images = images_seen
        images_seen += images.shape[0]
        pbar.update(images.shape[0])

        if crossed(previous_images, images_seen, cfg.log_every_steps):
            avg = {k: v / running_n for k, v in running.items()}
            postfix = {k: f"{v:.4f}" for k, v in avg.items()}
            postfix["lr"] = f"{lr:.2e}"
            pbar.set_postfix(postfix)
            # x-axis in images, not steps, so curves from runs at different
            # batch sizes lie on top of each other instead of being stretched
            # apart by a factor of batch.
            for k, v in avg.items():
                tb_writer.add_scalar(f"train/{k}", v, images_seen)
            tb_writer.add_scalar("train/lr", lr, images_seen)
            tb_writer.add_scalar("train/global_step", global_step, images_seen)
            running, running_n = {}, 0

        if crossed(previous_images, images_seen, cfg.eval_every_steps):
            evaluate(vqgan, eval_val_crops, device, out_dir, images_seen, preview_images,
                     tb_writer, cfg.batch_size, amp)
            vqgan.train()

        if crossed(previous_images, images_seen, cfg.checkpoint_every_steps):
            save_checkpoint(vqgan, discriminator, opt_g, opt_d, model_config,
                            global_step, images_seen, ema_switched, checkpoint_dir)

    pbar.close()
    save_checkpoint(vqgan, discriminator, opt_g, opt_d, model_config,
                    global_step, images_seen, ema_switched, checkpoint_dir, tag="last")
    tb_writer.close()


@torch.no_grad()
def evaluate(vqgan, eval_val_crops, device, out_dir, images_seen, preview_images, tb_writer,
             batch_size, amp=False):
    """Progress readout on a fixed, size-balanced subset of the validation
    split (see VQGANTrainConfig.eval_images/eval_size_groups and
    vqgan/data/eval_subset.py) — not what the validation set is: scripts/
    evaluate.py walks the split in full. `eval_val_crops` and `preview_images`
    are decoded once in main() and reused every call, so the curve is
    comparable to itself across evaluation points in this run.

    Run under the same bf16 autocast as training. In fp32 this was 2.6x more
    expensive for no useful precision: 2,048 crops took 79.6s against 30.3s,
    and the val L1 agreed to four decimals (0.1469 either way on the 40k
    checkpoint). Codebook usage does move a little — 99.1% vs 97.4% on that
    same pass, since rounding flips which code a few tokens land on — so read
    it as a health indicator, not an exact figure.
    """
    vqgan.eval()
    vqgan.quantizer.reset_usage_stats()

    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp)

    total_l1, n = 0.0, 0
    for i in range(0, eval_val_crops.shape[0], batch_size):
        images = eval_val_crops[i:i + batch_size].to(device)
        with autocast:
            recon = vqgan(images).recon.float()
        total_l1 += (recon - images).abs().mean().item() * images.shape[0]
        n += images.shape[0]
    val_l1 = total_l1 / max(n, 1)

    usage_pct = vqgan.quantizer.codebook_usage_pct()
    console.print(
        f"{images_seen:,} images: val L1 {val_l1:.4f}  codebook usage {usage_pct:.1f}%"
    )
    tb_writer.add_scalar("val/l1", val_l1, images_seen)
    tb_writer.add_scalar("val/codebook_usage_pct", usage_pct, images_seen)

    # preview_images is one representative image per size group and can run
    # well past batch_size crops (a large-group image may hold dozens), so
    # it gets the same chunked treatment as the main readout above.
    recons = []
    for i in range(0, preview_images.shape[0], batch_size):
        with autocast:
            recons.append(vqgan(preview_images[i:i + batch_size]).recon.float())
    recon = torch.cat(recons, dim=0)
    grid = make_grid(
        torch.cat([preview_images, recon], dim=0), nrow=min(preview_images.shape[0], 16)
    )
    grid = (grid + 1) / 2
    save_image(grid, out_dir / f"recon_img{images_seen:09d}.png")
    tb_writer.add_image("val/recon", grid, images_seen)


def save_checkpoint(
    vqgan, discriminator, opt_g, opt_d, model_config, global_step, images_seen, ema_switched,
    checkpoint_dir, tag=None,
):
    name = f"vqgan_{tag}" if tag else f"vqgan_step{global_step:07d}"
    path = checkpoint_dir / f"{name}.pt"
    ckpt = {
        "global_step": global_step,
        # What the schedule actually runs on. global_step is kept alongside it
        # for the filename and for reading old checkpoints, but two runs at
        # different batch sizes agree on images, not steps.
        "images_seen": images_seen,
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
