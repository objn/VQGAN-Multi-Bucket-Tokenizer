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

    # attach a refinement head to a trained backbone and train it alone
    python scripts/train_vqgan.py --vqgan-checkpoint checkpoints/vqgan_step0028582.pt \
        --refine-enabled true --refine-train-stage refine_only
"""

import argparse
import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqgan.config import DataConfig, VQGANTrainConfig
from vqgan.refine_config import RefineConfig
from vqgan.data import CropDataset, build_shards
from vqgan.data.eval_subset import (
    SelectedImage,
    build_balanced_val_subset,
    compute_log_size_edges,
    eval_image_floor,
    materialize_selected_crops,
)
from vqgan.data.sources import FolderShard, ParquetShard, count_parquet_rows
from vqgan.display import TrainingDisplay, console
from vqgan.models import VQGAN, PatchDiscriminator
from vqgan.training import cleanup_distributed, setup_distributed, train_step, unwrap


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


# The refinement head's parameters, by name, in every state_dict and
# named_parameters() this script walks. One constant, so the freezing, the
# optimizer split and the state-dict check below cannot drift apart.
REFINE_PREFIX = "decoder.refine."


def parse_args(argv=None):
    """-> (config, reset_discriminator, vqgan_checkpoint). Flags are derived
    from the dataclass fields, so adding a config knob adds its flag
    automatically; anything that is not a property of the run itself (like
    --reset-discriminator, which is about one resume) is added by hand and
    kept out of the config."""
    defaults = VQGANTrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)

    def add_flags(fields, prefix=""):
        for field, default in fields:
            flag = "--" + (prefix + field).replace("_", "-")
            if isinstance(default, bool):
                parser.add_argument(flag, type=lambda s: s.lower() != "false", default=default)
            else:
                parser.add_argument(flag, type=type(default), default=default)

    # A nested config block (RefineConfig) is not something the loop above can
    # derive a flag from, so its fields get their own pass under a
    # --<block>-<field> prefix. --refine-lr then stays visibly separate from
    # --lr, which is the point of keeping the block separate to begin with.
    nested = {name: value for name, value in defaults.__dict__.items()
              if dataclasses.is_dataclass(value)}
    add_flags([(n, v) for n, v in defaults.__dict__.items() if n not in nested])
    for name, block in nested.items():
        add_flags(block.__dict__.items(), prefix=f"{name}_")

    parser.add_argument(
        "--reset-discriminator", action="store_true",
        help="resume the generator but start the discriminator from scratch",
    )
    parser.add_argument(
        "--vqgan-checkpoint", default="",
        help="take the trained weights out of this checkpoint and start a fresh run around "
             "them - schedule back at image 0, and a new generator optimizer, since attaching "
             "a refinement head changes which parameters the generator has (the discriminator "
             "and its optimizer come across unchanged). This is how a head gets attached to a "
             "chosen backbone; --resume, which continues a run with its optimizer state and "
             "image count intact, is the other thing",
    )

    args = vars(parser.parse_args(argv))
    reset_discriminator = args.pop("reset_discriminator")
    vqgan_checkpoint = args.pop("vqgan_checkpoint")
    blocks = {
        name: type(block)(**{f: args.pop(f"{name}_{f}") for f in block.__dict__})
        for name, block in nested.items()
    }
    cfg = VQGANTrainConfig(**args, **blocks)

    if cfg.resume and vqgan_checkpoint:
        parser.error(
            "--resume and --vqgan-checkpoint both name a checkpoint to start from, but they "
            "mean different things: --resume continues that run (optimizer state, image "
            "count, EMA state), --vqgan-checkpoint keeps only its weights and starts a new "
            "run. Pass one or the other."
        )
    if cfg.refine.train_stage not in RefineConfig.STAGES:
        parser.error(
            f"--refine-train-stage must be one of {', '.join(RefineConfig.STAGES)}, "
            f"got {cfg.refine.train_stage!r}"
        )
    return cfg, reset_discriminator, vqgan_checkpoint


def merge_refine_config(model_config: dict, refine: RefineConfig, is_main: bool) -> dict:
    """The architecture to build, given the one stored in a checkpoint and the
    refinement head this run asks for.

    The backbone always comes from the checkpoint (see the resume block in
    main()), and so does the head - with the one exception that is the whole
    point of --refine-enabled: a checkpoint with no head, plus a run that wants
    one, is how a head gets attached to an already-trained backbone.

    The reverse is deliberately not symmetric. A checkpoint that carries a head
    keeps it even if this run never mentions one, because those weights are in
    the file and quietly dropping them would change what the model is;
    --refine-train-stage vq_only is how to train around it instead.
    """
    merged = dict(model_config)
    # Checkpoints written before the head existed carry none of these keys.
    merged.setdefault("refine_enabled", False)
    merged.setdefault("refine_hidden_channels", refine.hidden_channels)
    merged.setdefault("refine_num_blocks", refine.num_blocks)

    if refine.enabled and not merged["refine_enabled"]:
        merged.update(
            refine_enabled=True,
            refine_hidden_channels=refine.hidden_channels,
            refine_num_blocks=refine.num_blocks,
        )
        if is_main:
            console.print(
                f"[yellow]refine:[/yellow] attaching a new RefinementHead "
                f"({refine.hidden_channels} channels x {refine.num_blocks} blocks) to a "
                f"checkpoint that has none"
            )
    elif merged["refine_enabled"] and is_main:
        asked = (refine.hidden_channels, refine.num_blocks)
        stored = (merged["refine_hidden_channels"], merged["refine_num_blocks"])
        if refine.enabled and asked != stored:
            console.print(
                f"[yellow]refine:[/yellow] head shape comes from the checkpoint "
                f"({stored[0]} channels x {stored[1]} blocks), not the requested "
                f"({asked[0]} x {asked[1]}) - its weights are that shape"
            )
        elif not refine.enabled:
            console.print(
                "[yellow]refine:[/yellow] the checkpoint carries a RefinementHead, so it is "
                "kept and trained (--refine-train-stage vq_only trains around it)"
            )
    return merged


def apply_train_stage(vqgan, stage: str, *, ema_on: bool):
    """Freeze/unfreeze the generator for `stage` - see RefineConfig.train_stage.

    "joint" is a no-op in a fresh run, where everything requires grad already,
    but not after a stage change, which is why it is spelled out rather than
    skipped.

    `ema_on` is what stops this from undoing the EMA switch: once the quantizer
    updates its codebook from EMA buffers, that weight is out of the optimizer
    for good, and handing it a gradient back here would leave two mechanisms
    writing the same tensor.
    """
    for name, p in unwrap(vqgan).named_parameters():
        if name.startswith(REFINE_PREFIX):
            p.requires_grad_(stage != "vq_only")
        else:
            frozen_codebook = ema_on and name == "quantizer.codebook.weight"
            p.requires_grad_(stage != "refine_only" and not frozen_codebook)


def set_train_mode(vqgan, stage: str):
    """vqgan.train(), except that "refine_only" holds the quantizer in eval mode.

    The codebook's EMA update and its dead-code revival are gated on
    self.training, not on requires_grad, so clearing gradients alone would
    leave the codebook drifting underneath a head that is being trained
    against a backbone which is supposed to be standing still.

    One thing to read carefully because of this: with EMA updates off, the
    quantizer falls back to reporting its gradient-mode vq_loss (codebook +
    commitment rather than commitment alone), so the logged vq number jumps
    when a run crosses between refine_only and joint. Neither term reaches a
    frozen encoder or a codebook that is out of the optimizer, so it is a
    readout changing units, not a loss changing behaviour.
    """
    vqgan.train()
    if stage == "refine_only":
        unwrap(vqgan).quantizer.eval()


def build_opt_g(vqgan, base_lr: float, refine_lr: float):
    """The generator's Adam, over whatever is trainable at the moment.

    betas are a momentum window measured in steps, so unlike everything else
    here they do shift with batch size. Left fixed: (0.5, 0.9) is the pairing
    GAN training is known to be stable at, and compounding them per image the
    way the codebook EMA does would leave essentially no momentum at all at
    large batches (0.5 ** 16 is 1.5e-5).

    With `refine_lr` above 0 the head gets its own param group, tagged
    `refine_group` so the training loop can drive it with the head's own
    schedule (RefineConfig.lr and the three fields under it) instead of the
    generator's: a head attached to a backbone millions of images into its
    decay would otherwise be born at whatever near-min_lr rate that schedule
    has reached. `refine_lr` is the already-scaled rate — see
    VQGANTrainConfig.scaled_lr.
    """
    trainable = [(n, p) for n, p in unwrap(vqgan).named_parameters() if p.requires_grad]
    if refine_lr <= 0:
        groups = [{"params": [p for _, p in trainable]}]
    else:
        head = [p for n, p in trainable if n.startswith(REFINE_PREFIX)]
        rest = [p for n, p in trainable if not n.startswith(REFINE_PREFIX)]
        groups = [g for g in ({"params": rest},
                              {"params": head, "lr": refine_lr, "refine_group": True})
                  if g["params"]]
    return torch.optim.Adam(groups, lr=base_lr, betas=(0.5, 0.9))


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
    cfg, reset_discriminator, vqgan_checkpoint = parse_args(argv)
    torch.manual_seed(cfg.seed)
    # (False, 0, 0, 1) unless torchrun launched this, in which case every
    # count below that multiplies by world_size degenerates to what a single
    # process computed before DDP existed. See vqgan/training/distributed.py.
    is_distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"
    if device.type == "cuda":
        where = f"rank {rank}/{world_size} on " if is_distributed else ""
        print(f"Using GPU: {where}{torch.cuda.get_device_name(device)}")
    else:
        print("No GPU found, using CPU")

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
    # batch size — see VQGANTrainConfig.eval_images. batch_size * world_size is
    # the global batch (what the user asked for; each rank got a share of it),
    # which is the number that rationale is about.
    global_batch = cfg.batch_size * world_size
    eval_images = eval_image_floor(cfg.eval_images, global_batch)
    train_ds = CropDataset(
        build_shards(index, "train", cfg.source),
        tile_size=cfg.tile_size,
        tile_overlap_ratio=cfg.tile_overlap_ratio,
        shuffle_buffer=shuffle_buffer,
        augment=True,
        seed=cfg.seed,
        # Only the train stream is split across processes. val_ds below is
        # never iterated as a DataLoader — eval_subset.py reads its shard list
        # directly — and only rank 0 evaluates anyway.
        rank=rank,
        world_size=world_size,
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
    if is_main:
        console.print(f"[bold]source[/bold] {cfg.source} ({where}/)")
        for name, shards in (("train", train_ds.shards), ("validation", val_ds.shards)):
            console.print(f"  {name:>10}: {describe_shards(shards)}")

    # A fixed, size-balanced subset of the validation split for the eval
    # readout below — computed once here (header reads only), then decoded
    # once, so every evaluate() call for the rest of the run reuses the same
    # crops instead of re-scanning the split. See VQGANTrainConfig.eval_images,
    # eval_size_groups, eval_prep_file, and vqgan/data/eval_subset.py.
    #
    # Only rank 0 ever calls evaluate(), so only rank 0 builds any of this —
    # the others would be decoding thousands of crops to throw them away.
    eval_val_crops = preview_images = None
    if is_main:
        eval_params = {
            "source": cfg.source, "tile_size": cfg.tile_size,
            "tile_overlap_ratio": cfg.tile_overlap_ratio,
            "eval_images": eval_images, "eval_size_groups": cfg.eval_size_groups,
            "seed": cfg.seed,
        }
        if cfg.eval_prep_file:
            cache = torch.load(cfg.eval_prep_file, map_location="cpu")
            mismatched = {k: (cache["params"].get(k), v) for k, v in eval_params.items()
                          if cache["params"].get(k) != v}
            if mismatched:
                raise RuntimeError(
                    f"{cfg.eval_prep_file} was built with different settings than this run "
                    f"(cached, current): {mismatched} — rebuild it with 'Create prep data file' "
                    "so it matches, or clear eval_prep_file to build the subset fresh"
                )
            eval_val_crops, eval_offsets = cache["crops"], cache["offsets"]
            eval_selection = [SelectedImage(**d) for d in cache["selection"]]
            console.print(
                f"[bold]val readout[/bold] {eval_val_crops.shape[0]:,} crops loaded from "
                f"{cfg.eval_prep_file} (skipped the validation-split scan)"
            )
        else:
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
            floor_note = f" (above the {cfg.eval_images:,} floor, 64 x batch {global_batch})" \
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

    # Two flags, one file to read, and they differ in how much of it they
    # take. --resume continues a run: weights, optimizers, image count, EMA
    # state. --vqgan-checkpoint takes the weights alone and starts a new run
    # around them — which is what choosing a backbone to attach a refinement
    # head to needs, since the head changes the parameter set and the optimizer
    # state saved next to those weights no longer describes it. parse_args()
    # has already rejected both being passed at once.
    #
    # Either way the shape of the network is a fixed property of the file and
    # is read back from it rather than from cfg (whose defaults drift between
    # runs), otherwise load_state_dict below fails with a shape mismatch the
    # moment the two disagree. merge_refine_config() is the one seam where this
    # run gets a say: attaching a head the checkpoint does not have.
    checkpoint_path = cfg.resume or vqgan_checkpoint
    backbone_only = bool(vqgan_checkpoint)

    model_config = cfg.model_config()
    if checkpoint_path:
        resume_ckpt = torch.load(checkpoint_path, map_location=device)
        if "model_config" not in resume_ckpt:
            raise ValueError(
                f"{checkpoint_path} has no model_config entry — it predates the ViT-VQGAN "
                f"rewrite and holds CNN encoder/decoder weights that cannot be loaded. Move it "
                f"aside and train from scratch."
            )
        model_config = merge_refine_config(resume_ckpt["model_config"], cfg.refine, is_main)
        if model_config != cfg.model_config() and is_main:
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
    if is_main:
        console.print(f"generator: {n_params / 1e6:.1f}M params, token grid {grid_size}x{grid_size}")

    if resume_ckpt is not None:
        # A head this run is attaching for the first time is the one thing
        # allowed to be missing from the checkpoint. strict=False alone would
        # also wave through a genuinely mismatched checkpoint, so the keys it
        # let slide are checked by hand: anything missing outside the head, or
        # anything unexpected at all, is still an error.
        attaching_refine = (
            model_config["refine_enabled"]
            and REFINE_PREFIX + "in_conv.weight" not in resume_ckpt["vqgan"]
        )
        if attaching_refine:
            incompatible = vqgan.load_state_dict(resume_ckpt["vqgan"], strict=False)
            stray = [k for k in incompatible.missing_keys if not k.startswith(REFINE_PREFIX)]
            if stray or incompatible.unexpected_keys:
                raise ValueError(
                    f"{checkpoint_path} does not match this architecture — missing {stray}, "
                    f"unexpected {list(incompatible.unexpected_keys)}"
                )
            if is_main:
                console.print(
                    "[yellow]refine:[/yellow] RefinementHead started fresh (zero-init, so the "
                    "model reconstructs exactly as it did before until the head trains); "
                    "everything else loaded from the checkpoint"
                )
        else:
            vqgan.load_state_dict(resume_ckpt["vqgan"])
        if reset_discriminator:
            if is_main:
                console.print(
                    "[yellow]resume:[/yellow] discriminator reinitialized, generator kept"
                )
        else:
            try:
                discriminator.load_state_dict(resume_ckpt["discriminator"])
            except RuntimeError as e:
                raise ValueError(
                    f"{checkpoint_path} holds a discriminator this code cannot load: {e}\n"
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

        if backbone_only:
            # global_step and images_seen stay at 0: this is a new run, and its
            # schedule (lr decay, warmups, eval and checkpoint cadence) should
            # start from the beginning. ema_switched above is the exception —
            # it is a property of the codebook that came in the file, and
            # putting a converged codebook back through the gradient-update
            # warmup would only unsettle it.
            if is_main:
                console.print(
                    f"backbone loaded from {vqgan_checkpoint} (ema_switched={ema_switched}) — "
                    f"schedule and image count start from zero, generator optimizer fresh, "
                    f"discriminator and its optimizer carried over"
                )
        else:
            global_step = resume_ckpt["global_step"]
            # Checkpoints from before the schedule was counted in images only
            # know their step count, which meant reference_batch_size images each.
            images_seen = resume_ckpt.get(
                "images_seen", global_step * cfg.reference_batch_size
            )
            if is_main:
                console.print(
                    f"resumed from {cfg.resume} at {images_seen:,} images / step {global_step} "
                    f"(ema_switched={ema_switched})"
                )

    # Which half of the model trains. Applied before DDP wraps the model, since
    # DDP reads requires_grad when it builds its reducer, and before opt_g,
    # which is built from whatever is trainable once this has run.
    train_stage = cfg.refine.train_stage
    if train_stage == "refine_only" and not model_config["refine_enabled"]:
        raise ValueError(
            "--refine-train-stage refine_only has nothing to train: this run has no "
            "refinement head. Pass --refine-enabled true, with --vqgan-checkpoint <file> to "
            "attach one to an already-trained backbone."
        )
    if train_stage != "joint":
        apply_train_stage(vqgan, train_stage, ema_on=ema_switched)
        if is_main:
            trainable = sum(p.numel() for p in vqgan.parameters() if p.requires_grad)
            console.print(
                f"[bold]train stage[/bold] {train_stage} — {trainable / 1e6:.2f}M of "
                f"{n_params / 1e6:.1f}M generator params training"
            )
    if cfg.refine.warmup_steps > 0 and train_stage == "refine_only" and is_main:
        console.print(
            f"[bold]refine warmup[/bold] backbone unfreezes at "
            f"{cfg.refine.warmup_steps:,} images (refine_only -> joint)"
        )

    # DDP averages gradients across ranks, so the batch this rate is being
    # scaled for is the global one, not this rank's share — see scaled_lr().
    base_lr = cfg.scaled_lr(world_size)
    if base_lr != cfg.lr and is_main:
        console.print(
            f"[bold]lr[/bold] {base_lr:.2e} — {cfg.lr:.2e} scaled by {cfg.lr_scaling} "
            f"for batch {global_batch} vs reference {cfg.reference_batch_size}"
        )
    if cfg.end_steps_lr != cfg.max_steps and is_main:
        where = "before" if cfg.end_steps_lr < cfg.max_steps else "past"
        console.print(
            f"[bold]lr decay[/bold] reaches min_lr at {cfg.end_steps_lr:,} images, "
            f"{where} max_steps ({cfg.max_steps:,}) — "
            + ("lr sits flat at min_lr for the rest of the run"
               if cfg.end_steps_lr < cfg.max_steps
               else "lr never reaches min_lr within this run")
        )

    # Wrapped after any resume has been loaded, so DDP's constructor broadcasts
    # the resumed weights to every rank rather than a fresh random init.
    # broadcast_buffers=False because the one buffer set that has to agree
    # across ranks — the quantizer's EMA codebook — is synced explicitly in
    # VectorQuantizer.forward(); DDP's own per-forward broadcast would fight
    # with that by reinstating rank 0's copy over everyone's local update.
    if is_distributed:
        vqgan = DistributedDataParallel(
            vqgan, device_ids=[local_rank], broadcast_buffers=False
        )
        discriminator = DistributedDataParallel(
            discriminator, device_ids=[local_rank], broadcast_buffers=False
        )

    # The head's schedule, resolved once here rather than per step. Each field
    # reads 0/"" as "whatever the VQ side does" (see RefineConfig), which is
    # what `or` is spelling out below; with refine.lr itself at 0 none of it is
    # used at all and the head rides base_lr in the single param group.
    refine_lr_on = cfg.refine.lr > 0
    refine_base_lr = cfg.scaled_lr(
        world_size, lr=cfg.refine.lr, scaling=cfg.refine.lr_scaling or cfg.lr_scaling
    ) if refine_lr_on else 0.0
    refine_min_lr = cfg.refine.min_lr or cfg.min_lr
    refine_end_steps_lr = cfg.refine.end_steps_lr or cfg.end_steps_lr
    refine_lr_warmup_steps = cfg.refine.lr_warmup_steps or cfg.lr_warmup_steps

    opt_g = build_opt_g(vqgan, base_lr, refine_base_lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=base_lr, betas=(0.5, 0.9))
    if refine_lr_on and is_main:
        console.print(
            f"[bold]refine lr[/bold] {refine_base_lr:.2e} — {cfg.refine.lr:.2e} scaled by "
            f"{cfg.refine.lr_scaling or cfg.lr_scaling}, warming up over "
            f"{refine_lr_warmup_steps:,} images then decaying to {refine_min_lr:.2e} at "
            f"{refine_end_steps_lr:,.0f}, on its own curve from the generator's"
        )

    # opt_g is the only thing --vqgan-checkpoint cannot carry over: Adam's
    # state is per-parameter, and attaching a head changes which parameters the
    # generator has. opt_d is untouched by that — the refinement head is on the
    # generator side, PatchDiscriminator's shape is identical either way, and
    # the discriminator trains every step whatever train_stage says — so
    # resetting it would only throw away a trained adversary for no reason.
    if resume_ckpt is not None:
        if "opt_g" in resume_ckpt and not backbone_only:
            try:
                opt_g.load_state_dict(resume_ckpt["opt_g"])
            except ValueError as e:
                # --refine-train-stage and --refine-lr decide which parameters
                # go into opt_g and how they are grouped, and neither is stored
                # in the checkpoint — they describe a run, not its weights. So
                # a resume that changes either finds Adam's saved moments laid
                # out for a different optimizer, and says so in terms of the
                # flag that caused it rather than of param group sizes.
                saved = [len(g["params"]) for g in resume_ckpt["opt_g"]["param_groups"]]
                now = [len(g["params"]) for g in opt_g.param_groups]
                raise ValueError(
                    f"{cfg.resume} saved its generator optimizer over a different set of "
                    f"parameters than this run trains (param groups {saved} in the "
                    f"checkpoint, {now} here): {e}\n"
                    f"--refine-train-stage and --refine-lr are properties of a run and are "
                    f"not stored in the checkpoint, so a resume has to be given the same "
                    f"ones the checkpoint was written under (this run: "
                    f"train_stage={train_stage}, refine_lr={cfg.refine.lr}). Or pass "
                    f"--vqgan-checkpoint instead of --resume to keep the weights and start "
                    f"the generator optimizer fresh."
                ) from e
        # Its saved param_groups carry the lr it left off at, which for a run
        # starting again at image 0 is a decayed one — the cosine schedule
        # overwrites both optimizers' rates on every step below, so it never
        # gets used.
        if "opt_d" in resume_ckpt:
            opt_d.load_state_dict(resume_ckpt["opt_d"])

    # One writer and one tensorboard process for the run, not one per rank.
    tb_log_dir = out_dir / "tensorboard"
    tb_writer = None
    if is_main:
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        try:
            subprocess.Popen(["tensorboard", "--logdir", str(tb_log_dir), "--port", "6006"])
            console.print("[cyan]tensorboard:[/cyan] http://localhost:6006")
        except FileNotFoundError:
            console.print(
                f"[yellow]tensorboard CLI not found on PATH[/yellow] — logs are still written to {tb_log_dir}"
            )

    set_train_mode(vqgan, train_stage)
    discriminator.train()
    running = {}
    running_n = 0
    # Progress, logging and every trigger below are counted in images. Steps
    # are still counted, but only to name checkpoints and to average the
    # running loss over the batches that produced it.
    display = TrainingDisplay(total_images=cfg.max_steps, initial_images=images_seen) \
        if is_main else None
    if display is not None:
        display.start()

    for images in infinite(train_loader):
        if images_seen >= cfg.max_steps:
            break

        # Every rank flips at the same images_seen (the counter advances by the
        # same global amount everywhere), so the collectives inside the EMA
        # branch of the quantizer stay matched across ranks.
        if not ema_switched and images_seen >= cfg.ema_warmup_steps:
            unwrap(vqgan).quantizer.set_use_ema(True)
            opt_g = build_opt_g(vqgan, base_lr, refine_base_lr)
            ema_switched = True
            if is_main:
                console.print(
                    f"{images_seen:,} images: [bold cyan]switched quantizer to EMA mode[/bold cyan]"
                )

        # The head has had its solo run; let the backbone move again. Same
        # shape as the EMA switch above — a threshold in images, an optimizer
        # rebuilt around the parameter set that just changed, and a line in the
        # log saying where it happened. Every rank crosses it at the same
        # images_seen, so the freezing stays consistent across ranks.
        if (train_stage == "refine_only" and cfg.refine.warmup_steps > 0
                and images_seen >= cfg.refine.warmup_steps):
            train_stage = "joint"
            apply_train_stage(vqgan, train_stage, ema_on=ema_switched)
            opt_g = build_opt_g(vqgan, base_lr, refine_base_lr)
            set_train_mode(vqgan, train_stage)
            if is_main:
                console.print(
                    f"{images_seen:,} images: [bold cyan]unfroze the backbone[/bold cyan] "
                    f"(refine_only -> joint)"
                )

        images = images.to(device, non_blocking=True)

        lr = cosine_lr(images_seen, cfg.end_steps_lr, base_lr, cfg.min_lr, cfg.lr_warmup_steps)
        # Same counter, its own curve: ramp, decay and floor all come from the
        # head's own fields (see RefineConfig).
        refine_lr = cosine_lr(
            images_seen, refine_end_steps_lr, refine_base_lr, refine_min_lr,
            refine_lr_warmup_steps,
        ) if refine_lr_on else lr
        for group in opt_g.param_groups:
            # Only a run with its own refine rate has more than one group here.
            group["lr"] = refine_lr if group.get("refine_group") else lr
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
        # The whole run's schedule is counted in images, so this counts the
        # images *the run* consumed this step — every rank's batch, not just
        # this one's. No collective needed: drop_last=True makes every rank's
        # batch exactly cfg.batch_size, so the total is arithmetic. It also
        # keeps this counter identical on every rank, which is what keeps the
        # triggers below (and therefore the collectives they lead to) matched.
        step_images = images.shape[0] * world_size
        images_seen += step_images

        if is_main:
            display.advance(step_images)

        if crossed(previous_images, images_seen, cfg.log_every_steps):
            avg = {k: v / running_n for k, v in running.items()}
            if is_main:
                display.set_losses(avg, lr)
                # x-axis in images, not steps, so curves from runs at different
                # batch sizes lie on top of each other instead of being stretched
                # apart by a factor of batch.
                for k, v in avg.items():
                    tb_writer.add_scalar(f"train/{k}", v, images_seen)
                tb_writer.add_scalar("train/lr", lr, images_seen)
                tb_writer.add_scalar("train/global_step", global_step, images_seen)
            running, running_n = {}, 0

        # Rank 0 evaluates and writes; the others simply carry on to the next
        # step and block at its gradient all-reduce until rank 0 catches up.
        # evaluate() gets the unwrapped model so nothing touches DDP's
        # per-iteration bookkeeping outside the training step itself.
        if is_main and crossed(previous_images, images_seen, cfg.eval_every_steps):
            evaluate(unwrap(vqgan), eval_val_crops, device, out_dir, images_seen,
                     preview_images, tb_writer, cfg.batch_size, amp, display)
            set_train_mode(vqgan, train_stage)

        if is_main and crossed(previous_images, images_seen, cfg.checkpoint_every_steps):
            save_checkpoint(vqgan, discriminator, opt_g, opt_d, model_config,
                            global_step, images_seen, ema_switched, checkpoint_dir)

    if is_main:
        display.stop()
        save_checkpoint(vqgan, discriminator, opt_g, opt_d, model_config,
                        global_step, images_seen, ema_switched, checkpoint_dir, tag="last")
        tb_writer.close()
    cleanup_distributed(is_distributed)


@torch.no_grad()
def evaluate(vqgan, eval_val_crops, device, out_dir, images_seen, preview_images, tb_writer,
             batch_size, amp=False, display=None):
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

    # Usage answers "how many codes are ever touched", perplexity answers "how
    # many are actually carrying the representation" — a codebook can score
    # ~100% on the first while a handful of codes take nearly every lookup.
    # See VectorQuantizer.codebook_perplexity().
    quantizer = vqgan.quantizer
    usage_pct = quantizer.codebook_usage_pct()
    used_count = quantizer.codebook_used_count()
    perplexity = quantizer.codebook_perplexity()
    num_embeddings = quantizer.num_embeddings
    console.print(
        f"{images_seen:,} images: val L1 {val_l1:.4f}  "
        f"codebook {used_count:,}/{num_embeddings:,} used ({usage_pct:.1f}%)  "
        f"perplexity {perplexity:,.0f} ({100.0 * perplexity / num_embeddings:.1f}%)",
        soft_wrap=True,  # one eval per line, so the run's history stays greppable
    )
    tb_writer.add_scalar("val/l1", val_l1, images_seen)
    tb_writer.add_scalar("val/codebook_usage_pct", usage_pct, images_seen)
    tb_writer.add_scalar("val/codebook_used_count", used_count, images_seen)
    tb_writer.add_scalar("val/codebook_perplexity", perplexity, images_seen)
    tb_writer.add_scalar(
        "val/codebook_perplexity_ratio", perplexity / num_embeddings, images_seen
    )
    if display is not None:
        display.set_codebook(used_count, num_embeddings, usage_pct, perplexity)

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
        # unwrap() so a DDP run writes the same plain keys a single-GPU run
        # does — DDP's own state_dict() would prefix everything with "module."
        # and no other script in this project knows how to read that.
        "vqgan": unwrap(vqgan).state_dict(),
        "discriminator": unwrap(discriminator).state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }
    torch.save(ckpt, path)
    console.print(f"[green]saved[/green] {path}")


if __name__ == "__main__":
    main()
