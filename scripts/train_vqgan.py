"""Stage 2: train the ViT-VQGAN (ViT encoder + quantizer + ViT decoder + CNN discriminator)."""

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
    """Linear warmup then cosine decay to min_lr, in images. See doc/schedule-units.md"""
    if warmup_images > 0 and images_seen < warmup_images:
        return base_lr * images_seen / warmup_images
    progress = min(
        (images_seen - warmup_images) / max(1, end_images - warmup_images - 1), 1.0
    )
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def crossed(previous: int, current: int, every: int) -> bool:
    """Did the count pass a multiple of `every` between these readings? See doc/schedule-units.md"""
    return every > 0 and current // every > previous // every


# See doc/cli-flags.md:42 (refine_prefix)
REFINE_PREFIX = "decoder.refine."


def parse_args(argv=None):
    """-> (config, reset_discriminator, vqgan_checkpoint). See doc/cli-flags.md"""
    defaults = VQGANTrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)

    def add_flags(fields, prefix=""):
        for field, default in fields:
            flag = "--" + (prefix + field).replace("_", "-")
            if isinstance(default, bool):
                parser.add_argument(flag, type=lambda s: s.lower() != "false", default=default)
            else:
                parser.add_argument(flag, type=type(default), default=default)

    # See doc/cli-flags.md:18 (nested-blocks)
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
    # Here rather than in scaled_lr(), which raises too late to be useful
    if cfg.refine.lr_scaling not in ("sqrt", "linear", "none"):
        parser.error(
            f"--refine-lr-scaling must be sqrt, linear or none, got "
            f"{cfg.refine.lr_scaling!r}"
        )
    return cfg, reset_discriminator, vqgan_checkpoint


def merge_refine_config(model_config: dict, refine: RefineConfig, is_main: bool) -> dict:
    """The checkpoint's model_config, plus a head if this run is attaching one. See doc/resume-and-head.md"""
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
    """Freeze/unfreeze the generator for `stage`. See doc/resume-and-head.md"""
    for name, p in unwrap(vqgan).named_parameters():
        if name.startswith(REFINE_PREFIX):
            p.requires_grad_(stage != "vq_only")
        else:
            frozen_codebook = ema_on and name == "quantizer.codebook.weight"
            p.requires_grad_(stage != "refine_only" and not frozen_codebook)


def set_train_mode(vqgan, stage: str):
    """vqgan.train(), except that "refine_only" holds the quantizer in eval. See doc/resume-and-head.md"""
    vqgan.train()
    if stage == "refine_only":
        unwrap(vqgan).quantizer.eval()


def build_opt_g(vqgan, base_lr: float, refine_lr: float):
    """The generator's Adam, over every generator parameter. See doc/optimizer-state.md"""
    named = list(unwrap(vqgan).named_parameters())
    head = [p for n, p in named if n.startswith(REFINE_PREFIX)]
    rest = [p for n, p in named if not n.startswith(REFINE_PREFIX)]
    # Adam rejects an empty group, so a headless model yields one group
    groups = [g for g in ({"params": rest},
                          {"params": head, "lr": refine_lr, "refine_group": True})
              if g["params"]]
    return torch.optim.Adam(groups, lr=base_lr, betas=(0.5, 0.9))


def flat_named_params(vqgan):
    """Generator parameters as (name, param), in build_opt_g's group order."""
    named = list(unwrap(vqgan).named_parameters())
    head = [(n, p) for n, p in named if n.startswith(REFINE_PREFIX)]
    rest = [(n, p) for n, p in named if not n.startswith(REFINE_PREFIX)]
    return rest + head


def _state_fits(state: dict, candidates) -> bool:
    """Would `state`, read positionally, fit every one of `candidates`?"""
    if any(int(i) >= len(candidates) for i in state):
        return False
    return all(
        state[i]["exp_avg"].shape == candidates[int(i)][1].shape
        and state[i]["exp_avg_sq"].shape == candidates[int(i)][1].shape
        for i in state
    )


def adopt_opt_g_state(opt_g, saved: dict, flat) -> bool:
    """Load `saved` into `opt_g`, re-seating its state by name; -> whether it fit. See doc/optimizer-state.md"""
    state = {int(i): v for i, v in (saved.get("state") or {}).items()}
    if not state:
        return True

    names = saved.get("param_names")
    if names is not None:
        position = {n: i for i, (n, _) in enumerate(flat)}
        remapped = {
            position[names[i]]: v
            for i, v in state.items()
            if i < len(names) and names[i] in position
        }
    else:
        remapped = None
        for excluded in ((), ("quantizer.codebook.weight",)):
            candidates = [(n, p) for n, p in flat if n not in excluded]
            if not _state_fits(state, candidates):
                continue
            position = {n: i for i, (n, _) in enumerate(flat)}
            remapped = {position[candidates[i][0]]: v for i, v in state.items()}
            break
        if remapped is None:
            return False

    # See doc/optimizer-state.md:78 (re-seating-state-on-load)
    opt_g.load_state_dict({
        "state": remapped,
        "param_groups": opt_g.state_dict()["param_groups"],
    })
    return True


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
    # See doc/distributed.md
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
    # Measured against the batch it feeds, not in crops alone
    shuffle_buffer = max(cfg.shuffle_buffer, 8 * cfg.batch_size)
    # Scales with the machine; see VQGANTrainConfig.eval_images and
    # doc/eval-readout.md:6 (the-evaluation-subset). batch_size * world_size is global
    global_batch = cfg.batch_size * world_size
    eval_images = eval_image_floor(cfg.eval_images, global_batch)
    train_ds = CropDataset(
        build_shards(index, "train", cfg.source),
        tile_size=cfg.tile_size,
        tile_overlap_ratio=cfg.tile_overlap_ratio,
        shuffle_buffer=shuffle_buffer,
        augment=True,
        seed=cfg.seed,
        # See doc/distributed.md:9 (what-is-split-and-what-is-not)
        rank=rank,
        world_size=world_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        drop_last=True, pin_memory=True,
    )

    # See doc/eval-readout.md:6 (the-evaluation-subset)
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

    # The fixed val subset, built once and on rank 0 only.
    # See doc/eval-readout.md:6 (the-evaluation-subset) and vqgan/data/eval_subset.py
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

        # One representative image per size group. See doc/eval-readout.md:6 (the-evaluation-subset)
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
    # Three clocks: the run's, the backbone's, the head's. Each schedule
    # reads the clock of what it schedules. See doc/clocks.md
    vq_images_seen = 0
    refine_images_seen = 0
    # The head's optimizer steps, for the filename. See doc/clocks.md
    refine_global_step = 0
    ema_switched = False
    resume_ckpt = None

    # --resume continues a run; --vqgan-checkpoint takes the weights
    # alone. Architecture always comes from the file, and
    # merge_refine_config() is the only say this run gets.
    # See doc/resume-and-head.md
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

    # A property of the model, not of train_stage. See doc/resume-and-head.md
    head_present = model_config["refine_enabled"]

    # Gradient updates first, EMA after ema_warmup_steps
    vqgan = VQGAN(**model_config, use_ema=False).to(device)
    discriminator = PatchDiscriminator().to(device)
    grid_size = model_config["image_size"] // model_config["patch_size"]
    n_params = sum(p.numel() for p in vqgan.parameters())
    if is_main:
        console.print(f"generator: {n_params / 1e6:.1f}M params, token grid {grid_size}x{grid_size}")

    # Read again when opt_g is restored, below
    attaching_refine = False

    if resume_ckpt is not None:
        # A head being attached is the only key allowed to be missing, and
        # strict=False is checked by hand. See doc/resume-and-head.md
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

        # Directly, not via set_use_ema(). See doc/resume-and-head.md:90 (ema-state-on-resume)
        ema_switched = resume_ckpt.get("ema_switched", False)
        if ema_switched:
            vqgan.quantizer.use_ema = True
            vqgan.quantizer.codebook.weight.requires_grad_(False)

        if backbone_only:
            # The schedule restarts; ema_switched does not. See doc/resume-and-head.md
            if is_main:
                console.print(
                    f"backbone loaded from {vqgan_checkpoint} (ema_switched={ema_switched}) — "
                    f"schedule and image count start from zero, generator optimizer fresh, "
                    f"discriminator and its optimizer carried over"
                )
        else:
            global_step = resume_ckpt["global_step"]
            # Pre-images checkpoints know only their step count
            images_seen = resume_ckpt.get(
                "images_seen", global_step * cfg.reference_batch_size
            )
            # Defaults that read pre-split checkpoints right. See doc/clocks.md:107 (persistence)
            vq_images_seen = resume_ckpt.get("vq_images_seen", images_seen)
            if attaching_refine or not head_present:
                refine_images_seen = 0
                refine_global_step = 0
            else:
                refine_images_seen = resume_ckpt.get("refine_images_seen", images_seen)
                refine_global_step = resume_ckpt.get("refine_global_step", global_step)
            if is_main:
                console.print(
                    f"resumed from {cfg.resume} at {images_seen:,} images / step {global_step} "
                    f"(ema_switched={ema_switched})"
                )
                if head_present:
                    console.print(
                        f"[bold]clocks[/bold] backbone {vq_images_seen:,} images, head "
                        f"{refine_images_seen:,} images"
                        f"{' (new — its schedule starts here)' if attaching_refine else ''}"
                    )

    # Before DDP wraps and before opt_g is built. See doc/resume-and-head.md:56 (train-stages)
    train_stage = cfg.refine.train_stage
    if train_stage == "refine_only" and not model_config["refine_enabled"]:
        raise ValueError(
            "--refine-train-stage refine_only has nothing to train: this run has no "
            "refinement head. Pass --refine-enabled true to attach one — with --resume "
            "<file> to add it to the model and keep its image count, or with "
            "--vqgan-checkpoint <file> to take the weights alone and start a new run."
        )
    if train_stage != "joint":
        apply_train_stage(vqgan, train_stage, ema_on=ema_switched)
        if is_main:
            trainable = sum(p.numel() for p in vqgan.parameters() if p.requires_grad)
            console.print(
                f"[bold]train stage[/bold] {train_stage} — {trainable / 1e6:.2f}M of "
                f"{n_params / 1e6:.1f}M generator params training"
            )
    if cfg.refine.unfreeze_steps > 0 and train_stage == "refine_only" and is_main:
        console.print(
            f"[bold]refine unfreeze[/bold] backbone unfreezes once the head has had "
            f"{cfg.refine.unfreeze_steps:,} images (refine_only -> joint)"
        )

    # The global batch, not this rank's share. See doc/schedule-units.md:63 (batch-size-scaling-of-the-rate-itself)
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

    if head_present and cfg.refine.end_steps_lr != cfg.refine.max_steps and is_main:
        where = "before" if cfg.refine.end_steps_lr < cfg.refine.max_steps else "past"
        console.print(
            f"[bold]refine lr decay[/bold] reaches min_lr at "
            f"{cfg.refine.end_steps_lr:,.0f} head images, {where} the head's budget "
            f"({cfg.refine.max_steps:,}) — "
            + ("its rate sits flat at min_lr for the rest of the run"
               if cfg.refine.end_steps_lr < cfg.refine.max_steps
               else "its rate never reaches min_lr, so the run ends mid-decay")
        )

    # After the resume, so DDP broadcasts resumed weights, and with
    # broadcast_buffers off. See doc/distributed.md:20 (wrapping-order)
    if is_distributed:
        vqgan = DistributedDataParallel(
            vqgan, device_ids=[local_rank], broadcast_buffers=False
        )
        discriminator = DistributedDataParallel(
            discriminator, device_ids=[local_rank], broadcast_buffers=False
        )

    # One binding for every loss/eval setting, never merged.
    # See doc/resume-and-head.md:103 (which-config-the-run-is-trained-by)
    knobs = cfg.refine if head_present else cfg

    # Scaled once here; the rest of its curve is read per step
    refine_base_lr = cfg.scaled_lr(
        world_size, lr=cfg.refine.lr, scaling=cfg.refine.lr_scaling
    )

    opt_g = build_opt_g(vqgan, base_lr, refine_base_lr)
    # Position i in opt_g's state is opt_g_flat[i]. See doc/optimizer-state.md
    opt_g_flat = flat_named_params(vqgan)
    opt_g_names = [n for n, _ in opt_g_flat]
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=base_lr, betas=(0.5, 0.9))
    if head_present and is_main:
        console.print(
            f"[bold]refine lr[/bold] {refine_base_lr:.2e} — {cfg.refine.lr:.2e} scaled by "
            f"{cfg.refine.lr_scaling}, warming up over {cfg.refine.lr_warmup_steps:,} images "
            f"then decaying to {cfg.refine.min_lr:.2e} at {cfg.refine.end_steps_lr:,.0f} — "
            f"its own curve, unaffected by --lr and the rest of the VQ schedule"
        )
        # These are the --refine-* values, not the plain ones
        console.print(
            f"[bold]refine losses[/bold] l2 {knobs.l2_weight} laplace "
            f"{knobs.logit_laplace_weight} lpips {knobs.lpips_weight}"
            f"{'' if knobs.use_lpips else ' (off)'}, disc {knobs.disc_weight} after "
            f"{knobs.disc_warmup_steps:,} images, clip {knobs.grad_clip_norm}, eval every "
            f"{knobs.eval_every_steps:,}"
        )

    # opt_g comes across on any --resume, whatever stage either side was
    # in, re-seated by name. opt_d is never affected.
    # See doc/optimizer-state.md
    if resume_ckpt is not None:
        if "opt_g" in resume_ckpt and not backbone_only:
            if adopt_opt_g_state(opt_g, resume_ckpt["opt_g"], opt_g_flat):
                if attaching_refine and is_main:
                    console.print(
                        "[yellow]refine:[/yellow] the backbone's optimizer state came "
                        "across unchanged; the head starts with none of its own"
                    )
            else:
                saved = [len(g["params"]) for g in resume_ckpt["opt_g"]["param_groups"]]
                now = [len(g["params"]) for g in opt_g.param_groups]
                raise ValueError(
                    f"{cfg.resume} holds a generator optimizer whose state cannot be "
                    f"matched to this model's parameters (param groups {saved} in the "
                    f"checkpoint, {now} here, and no param_names to go by). A stage or "
                    f"EMA difference is carried across; this is a checkpoint from a "
                    f"different architecture. Pass --vqgan-checkpoint instead of "
                    f"--resume to keep the weights and start the optimizer fresh."
                )
        # Its saved rates are overwritten by the schedule every step
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
    # The bar tracks the budget that ends the run, on that budget's own
    # clock. See doc/clocks.md:68 (the-progress-bar)
    if train_stage == "refine_only":
        bar_total, bar_start = cfg.refine.max_steps, refine_images_seen
    else:
        bar_total, bar_start = cfg.max_steps, vq_images_seen
    display = TrainingDisplay(total_images=bar_total, initial_images=bar_start) \
        if is_main else None
    if display is not None:
        display.start()

    # What is already on disk for this model, so a run doing no steps
    # writes nothing. See doc/checkpoints.md:57 (the-final-save)
    last_saved_step = global_step

    for images in infinite(train_loader):
        # Each budget against its own clock; first one spent ends the run.
        # See doc/clocks.md:46 (budgets)
        if vq_images_seen >= cfg.max_steps:
            break
        if head_present and refine_images_seen >= cfg.refine.max_steps:
            break

        # On the backbone's clock, and with no optimizer rebuild.
        # See doc/clocks.md:91 (stage-transitions-read-the-right-clock) and
        # doc/optimizer-state.md:61 (no-mid-run-rebuilds)
        if not ema_switched and vq_images_seen >= cfg.ema_warmup_steps:
            unwrap(vqgan).quantizer.set_use_ema(True)
            ema_switched = True
            if is_main:
                console.print(
                    f"{images_seen:,} images: [bold cyan]switched quantizer to EMA mode[/bold cyan]"
                )

        # The head has had its solo run. On the head's clock, and with no
        # optimizer rebuild. See doc/clocks.md:91 (stage-transitions-read-the-right-clock)
        if (train_stage == "refine_only" and cfg.refine.unfreeze_steps > 0
                and refine_images_seen >= cfg.refine.unfreeze_steps):
            train_stage = "joint"
            apply_train_stage(vqgan, train_stage, ema_on=ema_switched)
            set_train_mode(vqgan, train_stage)
            if is_main:
                console.print(
                    f"{refine_images_seen:,} head images: [bold cyan]unfroze the "
                    f"backbone[/bold cyan] (refine_only -> joint)"
                )

        images = images.to(device, non_blocking=True)

        lr = cosine_lr(
            vq_images_seen, cfg.end_steps_lr, base_lr, cfg.min_lr, cfg.lr_warmup_steps
        )
        # Its own clock as well as its own curve. See doc/clocks.md
        refine_lr = cosine_lr(
            refine_images_seen, cfg.refine.end_steps_lr, refine_base_lr, cfg.refine.min_lr,
            cfg.refine.lr_warmup_steps,
        )
        for group in opt_g.param_groups:
            # Only a run with its own refine rate has more than one group here.
            group["lr"] = refine_lr if group.get("refine_group") else lr
        for group in opt_d.param_groups:
            group["lr"] = lr

        logs = train_step(
            vqgan, discriminator, opt_g, opt_d, images,
            l2_weight=knobs.l2_weight,
            logit_laplace_weight=knobs.logit_laplace_weight,
            lpips_weight=knobs.lpips_weight,
            disc_weight=knobs.disc_weight,
            use_lpips=knobs.use_lpips,
            # Paired with knobs: the head's warmup counts the head's images
            images_seen=refine_images_seen if head_present else vq_images_seen,
            disc_start_images=knobs.disc_warmup_steps,
            amp=amp,
            grad_clip_norm=knobs.grad_clip_norm,
        )
        for k, v in logs.items():
            if v is not None:
                running[k] = running.get(k, 0.0) + v
        running_n += 1
        global_step += 1
        previous_images = images_seen
        # Every rank's batch, and identical on every rank.
        # See doc/clocks.md:79 (counting-under-ddp)
        step_images = images.shape[0] * world_size
        images_seen += step_images
        # train_stage is re-read because the unfreeze above can change it
        if train_stage != "refine_only":
            vq_images_seen += step_images
        if head_present and train_stage != "vq_only":
            refine_images_seen += step_images
            refine_global_step += 1

        if is_main:
            # Same amount whichever clock the bar is drawn against
            display.advance(step_images)

        if crossed(previous_images, images_seen, cfg.log_every_steps):
            avg = {k: v / running_n for k, v in running.items()}
            if is_main:
                # Only the rates that belong to parameters with gradients.
                # See doc/eval-readout.md:90 (only-live-rates-are-shown)
                vq_training = train_stage != "refine_only"
                head_training = head_present and train_stage != "vq_only"
                # Each rate travels with the clock it was read against
                rates = []
                if vq_training:
                    rates.append(("vq", lr, vq_images_seen))
                if head_training:
                    rates.append(("head", refine_lr, refine_images_seen))
                display.set_losses(avg, rates)
                # x-axis in images so runs at different batches overlay
                for k, v in avg.items():
                    tb_writer.add_scalar(f"train/{k}", v, images_seen)
                if vq_training:
                    tb_writer.add_scalar("train/lr", lr, images_seen)
                if head_training:
                    tb_writer.add_scalar("train/refine_lr", refine_lr, images_seen)
                # The clocks, to place a curve later. See doc/clocks.md:107 (persistence)
                tb_writer.add_scalar("train/global_step", global_step, images_seen)
                tb_writer.add_scalar("train/vq_images_seen", vq_images_seen, images_seen)
                if head_present:
                    tb_writer.add_scalar(
                        "train/refine_images_seen", refine_images_seen, images_seen
                    )
            running, running_n = {}, 0

        # Rank 0 evaluates, unwrapped. See doc/distributed.md:9 (what-is-split-and-what-is-not)
        if is_main and crossed(previous_images, images_seen, knobs.eval_every_steps):
            evaluate(unwrap(vqgan), eval_val_crops, device, out_dir, images_seen,
                     preview_images, tb_writer, cfg.batch_size, amp, display)
            set_train_mode(vqgan, train_stage)

        if is_main and crossed(previous_images, images_seen, cfg.checkpoint_every_steps):
            save_checkpoint(vqgan, discriminator, opt_g, opt_d, model_config,
                            global_step, images_seen, ema_switched, checkpoint_dir,
                            vq_images_seen, refine_images_seen, refine_global_step,
                            opt_g_names)
            last_saved_step = global_step

    if is_main:
        display.stop()
        # Its own step number, never a fixed name, and skipped if the
        # periodic save already wrote this step. See doc/checkpoints.md:57 (the-final-save)
        if global_step != last_saved_step:
            save_checkpoint(vqgan, discriminator, opt_g, opt_d, model_config,
                            global_step, images_seen, ema_switched, checkpoint_dir,
                            vq_images_seen, refine_images_seen, refine_global_step,
                            opt_g_names)
        tb_writer.close()
    cleanup_distributed(is_distributed)


@torch.no_grad()
def evaluate(vqgan, eval_val_crops, device, out_dir, images_seen, preview_images, tb_writer,
             batch_size, amp=False, display=None):
    """Val L1, codebook health and a reconstruction preview. See doc/eval-readout.md:56 (the-display-rows)"""
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

    # Usage and perplexity answer different questions.
    # See doc/eval-readout.md:46 (codebook-metrics)
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

    # Can exceed batch_size crops, so chunked like the readout above
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
    checkpoint_dir, vq_images_seen, refine_images_seen, refine_global_step,
    opt_g_names,
):
    """Write one checkpoint, named for the steps it holds. See doc/checkpoints.md"""
    name = f"vqgan_step{global_step:07d}"
    if model_config["refine_enabled"]:
        name += f"_refine_step{refine_global_step:07d}"
    path = checkpoint_dir / f"{name}.pt"
    opt_g_state = opt_g.state_dict()
    # What Adam's positional keys refer to. See doc/optimizer-state.md:78 (re-seating-state-on-load)
    opt_g_state["param_names"] = list(opt_g_names)
    ckpt = {
        "global_step": global_step,
        # What the schedule runs on. See doc/schedule-units.md
        "images_seen": images_seen,
        # The training clocks; not recoverable afterwards. See doc/clocks.md:107 (persistence)
        "vq_images_seen": vq_images_seen,
        "refine_images_seen": refine_images_seen,
        "refine_global_step": refine_global_step,
        "ema_switched": ema_switched,
        "model_config": model_config,
        # Plain keys even under DDP. See doc/distributed.md:72 (writing-checkpoints)
        "vqgan": unwrap(vqgan).state_dict(),
        "discriminator": unwrap(discriminator).state_dict(),
        "opt_g": opt_g_state,
        "opt_d": opt_d.state_dict(),
    }
    torch.save(ckpt, path)
    console.print(f"[green]saved[/green] {path}")


if __name__ == "__main__":
    main()
