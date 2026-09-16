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
    python scripts/train_vqgan.py --resume checkpoints/vqgan_step0028582.pt

    # add a refinement head to the model and train it alone, keeping the image
    # count and the schedule the checkpoint arrived with
    python scripts/train_vqgan.py --resume checkpoints/vqgan_step0028582.pt \
        --refine-enabled true --refine-train-stage refine_only

    # the same head, but on a new run whose schedule restarts at image 0
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
    # Checked here rather than left to scaled_lr(), which would not raise until
    # the run is already several seconds into building a model.
    if cfg.refine.lr_scaling not in ("sqrt", "linear", "none"):
        parser.error(
            f"--refine-lr-scaling must be sqrt, linear or none, got "
            f"{cfg.refine.lr_scaling!r}"
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
    """The generator's Adam, over every generator parameter.

    betas are a momentum window measured in steps, so unlike everything else
    here they do shift with batch size. Left fixed: (0.5, 0.9) is the pairing
    GAN training is known to be stable at, and compounding them per image the
    way the codebook EMA does would leave essentially no momentum at all at
    large batches (0.5 ** 16 is 1.5e-5).

    Two param groups, always, and the same parameters in each whatever
    train_stage is: the head, tagged `refine_group` so the training loop can
    drive it from RefineConfig's schedule, and the rest of the generator on
    its own. They are two independent schedules over two clocks — see
    RefineConfig. `refine_lr` is the head's already-scaled rate (see
    VQGANTrainConfig.scaled_lr).

    Note what is *not* filtered here: parameters a stage has frozen go in
    anyway. Adam skips any parameter whose `.grad` is None, so a frozen one is
    held without being stepped and without so much as an exp_avg allocated for
    it — freezing is enforced by requires_grad, not by optimizer membership,
    and the two say the same thing.

    Handing it the frozen ones buys a stable layout. Adam's state is keyed by
    each parameter's *position* in these groups, so filtering made the key
    space a function of train_stage: index 0 was the encoder's position
    embedding in a joint run and the head's first convolution in a refine_only
    one, and a saved optimizer could not be read back by a run in a different
    stage — a resume across stages failed, and a refine_only run dropped the
    backbone's moments on the floor rather than carrying them through. With
    every parameter present the positions mean the same thing in every stage,
    so the state loads, survives a refine run untouched, and is still there
    when the backbone starts training again.

    The cost is that Adam walks the frozen parameters each step to skip them.
    """
    named = list(unwrap(vqgan).named_parameters())
    head = [p for n, p in named if n.startswith(REFINE_PREFIX)]
    rest = [p for n, p in named if not n.startswith(REFINE_PREFIX)]
    # Empty only when the model has no head at all; Adam rejects an empty
    # group, and a model without a head has no second schedule to drive.
    groups = [g for g in ({"params": rest},
                          {"params": head, "lr": refine_lr, "refine_group": True})
              if g["params"]]
    return torch.optim.Adam(groups, lr=base_lr, betas=(0.5, 0.9))


def flat_named_params(vqgan):
    """The generator's parameters as (name, param), in build_opt_g's group
    order — the order Adam numbers its state by."""
    named = list(unwrap(vqgan).named_parameters())
    head = [(n, p) for n, p in named if n.startswith(REFINE_PREFIX)]
    rest = [(n, p) for n, p in named if not n.startswith(REFINE_PREFIX)]
    return rest + head


def _state_fits(state: dict, candidates) -> bool:
    """Would `state`, read positionally, land a moment of the right shape on
    every one of `candidates`?"""
    if any(int(i) >= len(candidates) for i in state):
        return False
    return all(
        state[i]["exp_avg"].shape == candidates[int(i)][1].shape
        and state[i]["exp_avg_sq"].shape == candidates[int(i)][1].shape
        for i in state
    )


def adopt_opt_g_state(opt_g, saved: dict, flat) -> bool:
    """Load `saved` into `opt_g`, re-seating its state onto the right
    parameters, and return whether that could be done.

    Adam keys its state by each parameter's *position* in the optimizer's
    groups, and a position is only meaningful next to the list of parameters
    it was numbered against. Two things move it: adding a head (the group list
    grows), and — before build_opt_g stopped filtering — a stage or an EMA
    switch excluding something from the middle. The codebook leaving the
    optimizer once EMA takes over shifts every parameter after it by one, so
    reading the state positionally against today's full list would hand a
    hundred-odd parameters somebody else's moments. Silently: the shapes are
    wrong, but nothing checks them, and Adam would carry on with moments that
    describe a different tensor.

    So the state is re-seated by name. Checkpoints written from here on store
    `param_names` for exactly this, and names are matched to today's
    positions; anything the file has that this model does not is dropped, and
    anything new (a freshly attached head) simply starts without state.

    Checkpoints written before that carry no names, and their positions have
    to be reconstructed. Only the exclusions the old build_opt_g could
    actually produce are tried — it filtered on requires_grad, so the
    candidates are "everything" and "everything but the codebook" — and each
    is accepted only if every saved moment's shape matches the parameter it
    would land on. A file that fits neither is refused rather than guessed at.
    """
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

    # This run's group description, not the file's: it has the head's group,
    # and the rates on it are overwritten from the schedule every step anyway.
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
    # Three clocks, because "how far along is this run" and "how much training
    # has this half of the model actually had" stop being the same question the
    # moment a stage freezes something.
    #
    #   images_seen         the run's wall clock — every image pulled from the
    #                       loader, whatever it was used for. max_steps, the
    #                       eval/checkpoint/log cadence and the progress bar
    #                       are all measured on it, and it never stops.
    #   vq_images_seen      images that moved the backbone. Stands still under
    #                       refine_only. Drives the VQ lr curve and the EMA
    #                       switch.
    #   refine_images_seen  images that moved the refinement head, counted from
    #                       the moment the head was created. Stands still under
    #                       vq_only, and is 0 on a model that has no head.
    #                       Drives the head's lr curve, its discriminator
    #                       warmup and unfreeze_steps.
    #
    # Each schedule then reads the clock of the thing it schedules, so a warmup
    # measures the age of the parameters it is ramping rather than the age of
    # the run that happens to contain them.
    vq_images_seen = 0
    refine_images_seen = 0
    # Optimizer steps the head has taken. Not a schedule input — the second
    # half of the checkpoint filename, the counterpart of global_step.
    refine_global_step = 0
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

    # Whether this run's model has a refinement head — settled the moment
    # model_config is, and asked often enough below (the budget, the clocks,
    # the readout, the filename) to be worth a name rather than a dict lookup
    # each time. Note it is a property of the model, not of train_stage: a
    # vq_only run on a model with a head still has one.
    head_present = model_config["refine_enabled"]

    # Start with gradient-based codebook updates; switch to EMA after
    # ema_warmup_steps once the encoder has stabilized (EMA from image 0 can
    # lock in a noisy initial encoder).
    vqgan = VQGAN(**model_config, use_ema=False).to(device)
    discriminator = PatchDiscriminator().to(device)
    grid_size = model_config["image_size"] // model_config["patch_size"]
    n_params = sum(p.numel() for p in vqgan.parameters())
    if is_main:
        console.print(f"generator: {n_params / 1e6:.1f}M params, token grid {grid_size}x{grid_size}")

    # Set by the block below, and read again when the generator optimizer is
    # restored: attaching a head is the one resume that cannot carry opt_g over.
    attaching_refine = False

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
            # Checkpoints written before the clocks were split carry only
            # images_seen, and for them it is the right answer for the backbone:
            # nothing could freeze it, so every image it saw trained it.
            vq_images_seen = resume_ckpt.get("vq_images_seen", images_seen)
            # The head's clock, on the other hand, is 0 unless the file already
            # had a head — and if it had one without recording its age, that
            # head was trained by a run whose whole image count went to it.
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

    # Which half of the model trains. Applied before DDP wraps the model, since
    # DDP reads requires_grad when it builds its reducer, and before opt_g,
    # which is built from whatever is trainable once this has run.
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

    # Which config the loss, adversarial and eval settings come from. A model
    # with a refinement head is trained by RefineConfig's copies of them and a
    # model without one by VQGANTrainConfig's — the field names match exactly,
    # so the choice is this one binding rather than a fallback at each use.
    # Nothing is merged: picking a side picks all of it.
    knobs = cfg.refine if head_present else cfg

    # The head's rate, scaled once here the way base_lr was. Everything else
    # about its curve is read straight off cfg.refine at each step: none of it
    # falls back to the VQ schedule (see RefineConfig).
    refine_base_lr = cfg.scaled_lr(
        world_size, lr=cfg.refine.lr, scaling=cfg.refine.lr_scaling
    )

    opt_g = build_opt_g(vqgan, base_lr, refine_base_lr)
    # The same order build_opt_g put them in, so position i in opt_g's state is
    # opt_g_flat[i]. The names go into every checkpoint and the pairs are what
    # re-seats a resumed one — see adopt_opt_g_state.
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
        # Spelled out because these are the --refine-* values, not the plain
        # ones: a --disc-weight or --lpips-weight passed to this run was read
        # into a config it is not training from.
        console.print(
            f"[bold]refine losses[/bold] l2 {knobs.l2_weight} laplace "
            f"{knobs.logit_laplace_weight} lpips {knobs.lpips_weight}"
            f"{'' if knobs.use_lpips else ' (off)'}, disc {knobs.disc_weight} after "
            f"{knobs.disc_warmup_steps:,} images, clip {knobs.grad_clip_norm}, eval every "
            f"{knobs.eval_every_steps:,}"
        )

    # opt_g comes across on any --resume now, whatever stage either side was in.
    # build_opt_g holds every generator parameter rather than only the unfrozen
    # ones, so Adam's positional state keys mean the same thing in every stage:
    # a refine_only run loads the backbone's moments, leaves them untouched for
    # the length of the run (a frozen parameter is never stepped) and writes
    # them back out. They are still there when the backbone trains again, and
    # every checkpoint is the same size rather than shrinking to the third of
    # itself that a refine run happens to be using.
    #
    # Attaching a head is the one case where the *layout* changes — one group
    # becomes two — and adopt_opt_g_state() handles it by keeping the state and
    # taking this run's group description.
    #
    # --vqgan-checkpoint still starts fresh, by definition: it takes weights
    # alone and begins a new run around them.
    #
    # opt_d is untouched by any of this — the refinement head is on the
    # generator side, PatchDiscriminator's shape is identical either way, and
    # the discriminator trains every step whatever train_stage says — so
    # resetting it would only throw away a trained adversary for no reason.
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
    # The bar tracks the budget that will actually end the run, on that
    # budget's own clock. A refine_only run drawn against the VQ pair would
    # open at "1,400,004/3,400,000 — 41%" and finish at 43%, describing a
    # backbone that never moves, while the thing being trained went from 0
    # to done.
    if train_stage == "refine_only":
        bar_total, bar_start = cfg.refine.max_steps, refine_images_seen
    else:
        bar_total, bar_start = cfg.max_steps, vq_images_seen
    display = TrainingDisplay(total_images=bar_total, initial_images=bar_start) \
        if is_main else None
    if display is not None:
        display.start()

    # The step number already represented on disk for this model: whatever a
    # --resume arrived at, or nothing at all for a fresh run. The save at the
    # end of the run compares against it, so a run that does no steps — a
    # --resume of a checkpoint already at max_steps, which breaks out of the
    # loop immediately — writes nothing rather than rewriting the very file it
    # was started from under a fresh optimizer.
    last_saved_step = global_step

    for images in infinite(train_loader):
        # Each budget against its own clock. The backbone's is an absolute
        # position it has been walking towards since image 0; the head's is a
        # length, counted from the moment it was created. Neither is measured
        # on images_seen, which counts work done rather than progress along
        # either schedule — a refine_only run pushes it forward while the
        # backbone's position does not move at all.
        #
        # Whichever is spent first ends the run. Under refine_only only the
        # head's can be, since vq_images_seen is standing still, which is what
        # makes cfg.refine.max_steps the length of a refine run.
        if vq_images_seen >= cfg.max_steps:
            break
        if head_present and refine_images_seen >= cfg.refine.max_steps:
            break

        # Every rank flips at the same images_seen (the counter advances by the
        # same global amount everywhere), so the collectives inside the EMA
        # branch of the quantizer stay matched across ranks.
        # The backbone's clock, not the run's: the warmup exists to let the
        # encoder stabilize before the codebook starts following it, and a
        # refine_only stretch stabilizes nothing — the encoder is frozen and
        # the quantizer is in eval mode throughout.
        #
        # No optimizer rebuild here. set_use_ema() clears the codebook's
        # requires_grad so that EMA is the only thing writing it, and since
        # build_opt_g holds every parameter regardless, Adam simply stops
        # stepping one it is still holding. Rebuilding used to be what took the
        # codebook out of the optimizer, and it threw away every other
        # parameter's moments to do it.
        if not ema_switched and vq_images_seen >= cfg.ema_warmup_steps:
            unwrap(vqgan).quantizer.set_use_ema(True)
            ema_switched = True
            if is_main:
                console.print(
                    f"{images_seen:,} images: [bold cyan]switched quantizer to EMA mode[/bold cyan]"
                )

        # The head has had its solo run; let the backbone move again. Same
        # shape as the EMA switch above — a threshold in images and a line in
        # the log saying where it happened. Every rank crosses it at the same
        # refine_images_seen, so the freezing stays consistent across ranks.
        #
        # And as there, no optimizer rebuild: opt_g already holds the backbone,
        # so clearing requires_grad is the whole of it. The backbone picks its
        # moments up where the last run that trained it left off, and the head
        # keeps its own across the switch instead of being reset at the exact
        # moment it stops being the only thing training.
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
        # Its own clock as well as its own curve: rate, ramp, decay and floor
        # all come from the head's own fields, read against the head's own age
        # (see RefineConfig). This is what makes lr_warmup_steps mean anything
        # when a head is added to a model millions of images in — the ramp
        # starts where the head starts, not where the run does.
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
            # Paired with knobs above: a run training a head measures its
            # discriminator warmup in the head's images, so "let it settle for
            # 50,000 before it faces the adversary" counts the head's first
            # 50,000 and not a number the backbone passed long ago.
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
        # The whole run's schedule is counted in images, so this counts the
        # images *the run* consumed this step — every rank's batch, not just
        # this one's. No collective needed: drop_last=True makes every rank's
        # batch exactly cfg.batch_size, so the total is arithmetic. It also
        # keeps this counter identical on every rank, which is what keeps the
        # triggers below (and therefore the collectives they lead to) matched.
        step_images = images.shape[0] * world_size
        images_seen += step_images
        # Each training clock advances only while the half it measures is the
        # one being trained. train_stage is read fresh every step because the
        # unfreeze above can change it mid-run.
        if train_stage != "refine_only":
            vq_images_seen += step_images
        if head_present and train_stage != "vq_only":
            refine_images_seen += step_images
            refine_global_step += 1

        if is_main:
            # Same amount whichever clock the bar is drawn against: every
            # step feeds the whole batch to whatever is training, so that
            # clock advances by step_images exactly as images_seen does.
            display.advance(step_images)

        if crossed(previous_images, images_seen, cfg.log_every_steps):
            avg = {k: v / running_n for k, v in running.items()}
            if is_main:
                # Report the rate of whatever has gradients this step, and only
                # that. Under refine_only the generator's rate is still being
                # computed and still decaying, but it belongs to frozen
                # parameters — printing it next to the losses invites reading
                # the wrong curve to decide whether the schedule is working.
                vq_training = train_stage != "refine_only"
                head_training = head_present and train_stage != "vq_only"
                # Each rate travels with the clock it was read against, and
                # only the halves that have gradients get an entry — see
                # TrainingDisplay.set_losses.
                rates = []
                if vq_training:
                    rates.append(("vq", lr, vq_images_seen))
                if head_training:
                    rates.append(("head", refine_lr, refine_images_seen))
                display.set_losses(avg, rates)
                # x-axis in images, not steps, so curves from runs at different
                # batch sizes lie on top of each other instead of being stretched
                # apart by a factor of batch.
                for k, v in avg.items():
                    tb_writer.add_scalar(f"train/{k}", v, images_seen)
                if vq_training:
                    tb_writer.add_scalar("train/lr", lr, images_seen)
                if head_training:
                    tb_writer.add_scalar("train/refine_lr", refine_lr, images_seen)
                # The clocks, so a curve read later can be placed against the
                # age of the half of the model that produced it rather than
                # against the run that contained it.
                tb_writer.add_scalar("train/global_step", global_step, images_seen)
                tb_writer.add_scalar("train/vq_images_seen", vq_images_seen, images_seen)
                if head_present:
                    tb_writer.add_scalar(
                        "train/refine_images_seen", refine_images_seen, images_seen
                    )
            running, running_n = {}, 0

        # Rank 0 evaluates and writes; the others simply carry on to the next
        # step and block at its gradient all-reduce until rank 0 catches up.
        # evaluate() gets the unwrapped model so nothing touches DDP's
        # per-iteration bookkeeping outside the training step itself.
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
        # The end of the run gets a checkpoint under its own step number, not a
        # fixed vqgan_last.pt. A fixed name is a file every run overwrites, and
        # what it overwrites is whatever the run before it spent hours
        # producing — including, if that name was ever used as a starting
        # point, this run's own. Numbered files are written once and never
        # again, and "the newest" is just the highest number on disk
        # (vqgan.checkpoints.latest_step_checkpoint). Skipped when the periodic
        # save above already wrote this exact step.
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
    checkpoint_dir, vq_images_seen, refine_images_seen, refine_global_step,
    opt_g_names,
):
    """Write one checkpoint, named for the steps it holds.

    A model with no head is `vqgan_step0028582.pt`; one with a head is
    `vqgan_step0028582_refine_step0050000.pt`, the second number being the
    head's own step count. Both only ever move forward, so a name is claimed
    once and the file under it is never rewritten — and the name says, without
    being opened, whether the file has a head and how much training each half
    of it has had. See vqgan/checkpoints.py for the reading half.
    """
    name = f"vqgan_step{global_step:07d}"
    if model_config["refine_enabled"]:
        name += f"_refine_step{refine_global_step:07d}"
    path = checkpoint_dir / f"{name}.pt"
    opt_g_state = opt_g.state_dict()
    # What Adam's positional state keys refer to. Without it a reader has to
    # assume the parameter list has not changed shape since — which is exactly
    # the assumption that breaks across a stage or an EMA switch.
    opt_g_state["param_names"] = list(opt_g_names)
    ckpt = {
        "global_step": global_step,
        # What the schedule actually runs on. global_step is kept alongside it
        # for the filename and for reading old checkpoints, but two runs at
        # different batch sizes agree on images, not steps.
        "images_seen": images_seen,
        # The two training clocks (see main). Stored because they cannot be
        # recovered from images_seen — how much of a run went to each half
        # depends on the stages it passed through, which nothing else records.
        "vq_images_seen": vq_images_seen,
        "refine_images_seen": refine_images_seen,
        "refine_global_step": refine_global_step,
        "ema_switched": ema_switched,
        "model_config": model_config,
        # unwrap() so a DDP run writes the same plain keys a single-GPU run
        # does — DDP's own state_dict() would prefix everything with "module."
        # and no other script in this project knows how to read that.
        "vqgan": unwrap(vqgan).state_dict(),
        "discriminator": unwrap(discriminator).state_dict(),
        "opt_g": opt_g_state,
        "opt_d": opt_d.state_dict(),
    }
    torch.save(ckpt, path)
    console.print(f"[green]saved[/green] {path}")


if __name__ == "__main__":
    main()
