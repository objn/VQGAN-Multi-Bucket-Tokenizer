"""Interactive menu entrypoint for the ViT-VQGAN pipeline.

    python main.py

Each menu item is a thin wrapper around the corresponding scripts/*.py CLI —
for full control over every flag, call those scripts directly instead
(e.g. `python scripts/train_vqgan.py --max-steps 1600000 --batch-size 8`).

Autoregressive Transformer generation is out of scope for now — this project
is focused on getting ViT-VQGAN encode/decode reconstruction quality right first.
"""

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from scripts import (
    build_index,
    count_crops,
    evaluate,
    prep_eval_data,
    reconstruct,
    test_whole_image,
    train_vqgan,
    visualize_model,
)
from vqgan.config import DataConfig, VQGANTrainConfig
from vqgan.refine_config import RefineConfig
from vqgan.display import console

_STEP_CKPT_RE = re.compile(r"vqgan_step(\d+)\.pt")


def ask(prompt: str, default) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else str(default)


def latest_step_checkpoint(checkpoint_dir) -> str:
    """Newest vqgan_stepNNNNNNN.pt in checkpoint_dir by step number — never
    vqgan_last.pt, which gets overwritten every run and isn't tied to a
    specific step. Returns "" if none exist."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return ""
    candidates = []
    for p in checkpoint_dir.iterdir():
        m = _STEP_CKPT_RE.fullmatch(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return ""
    return str(max(candidates, key=lambda t: t[0])[1])


def run_build_index():
    """Index both corpora. The only real decision here is how to split the
    user's own images — the two source directories and the index path are
    fixed parts of the project layout, and asking for them would just invite
    setting one here and forgetting to match it in Train."""
    defaults = DataConfig()
    console.print(f"[dim]reading {defaults.manifest_dir}/ (dataset's own splits) "
                  f"and {defaults.folder_root}/[/dim]")
    console.print(f"[dim]the ratios below split {defaults.folder_root}/ only[/dim]")
    val_frac = ask("validation fraction", defaults.val_frac)
    test_frac = ask("test fraction", defaults.test_frac)
    build_index.main(["--val-frac", val_frac, "--test-frac", test_frac])


def run_count_crops():
    """How many tiles the current index yields — a read of data/index.json's
    headers only, no training or model involved."""
    defaults = VQGANTrainConfig()
    source = ask("Source (parquet/folder/all)", defaults.source)
    tile_size = ask("Tile size", defaults.tile_size)
    overlap = ask("Tile overlap ratio", defaults.tile_overlap_ratio)
    count_crops.main([
        "--source", source,
        "--tile-size", tile_size, "--tile-overlap-ratio", overlap,
    ])


def run_prep_eval_data():
    """Cache the validation eval subset (see vqgan/data/eval_subset.py) to
    disk once, so Train/Finetune can load it back instead of re-scanning the
    whole validation split at every startup. Rebuild it whenever source,
    tile_size, tile_overlap_ratio, eval_images, eval_size_groups, seed, or
    the batch size you'll train at changes — train_vqgan.py refuses a
    mismatched cache rather than silently evaluating on the wrong subset."""
    defaults = VQGANTrainConfig()
    source = ask("Source (parquet/folder/all)", defaults.source)
    batch_size = ask("Batch size you'll train at (sets the eval-size floor)", defaults.batch_size)
    out = ask("Output file", "data/eval_prep.pt")
    prep_eval_data.main(["--source", source, "--batch-size", batch_size, "--out", out])


def _launch_ddp(argv, num_gpus: int):
    """Run scripts/train_vqgan.py under torchrun, one process per GPU.

    DDP needs N operating-system processes, so this is the one menu action
    that shells out instead of calling into the script in-process. Every flag
    is the same one the single-GPU path passes — only --batch-size differs,
    carrying this rank's share rather than the total.
    """
    script = Path(__file__).resolve().parent / "scripts" / "train_vqgan.py"
    # --standalone: one box, no rendezvous endpoint to configure.
    command = ["torchrun", "--standalone", f"--nproc_per_node={num_gpus}", str(script), *argv]
    console.print(f"[dim]{' '.join(command)}[/dim]")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "torchrun was not found on PATH — it ships with torch, so this usually means the "
            "menu is running under a different interpreter than the one torch is installed in"
        ) from None


def _run_training(source: str, *, lr_default, require_checkpoint: bool):
    defaults = VQGANTrainConfig()
    # Asked first because it changes what the batch answer below *means*: with
    # DDP the number is the total across cards, so that "batch 64" describes
    # the same run whether it lands on one GPU or four.
    use_ddp = ask("Use DDP (multi-GPU)? (y/n)", "n")
    num_gpus = 1
    if use_ddp.strip().lower().startswith("y"):
        available = torch.cuda.device_count()
        num_gpus = int(ask("Number of GPUs", available))
        if num_gpus < 1:
            raise RuntimeError(f"need at least 1 GPU to train, got {num_gpus}")
        if num_gpus > available:
            # Caught here rather than as "invalid device ordinal" from the rank
            # that gets a card that isn't there, several screens of torchrun
            # traceback later.
            raise RuntimeError(
                f"asked for {num_gpus} GPUs but torch can only see {available} on this machine"
            )

    # No prompt for the index path: it is a fixed project location, and typing
    # a different one here without matching it in Build data index would
    # silently train on a stale index.
    # Named in steps (at batch_size=1, so also just images — see
    # VQGANTrainConfig's "Schedule" section) rather than images so it reads as
    # "how long to train": that number is batch-invariant, so changing the
    # batch below only changes how fast it gets there, not the answer.
    max_steps = ask("Max steps", defaults.max_steps)
    if num_gpus > 1:
        total_batch = int(ask(f"Batch size (total across {num_gpus} GPUs)", defaults.batch_size))
        if total_batch % num_gpus:
            # Refused rather than rounded: the schedule, the learning rate and
            # the eval readout are all derived from batch x GPUs, so silently
            # training at a different total than the one asked for would put
            # every one of them somewhere the user did not choose.
            floor = total_batch - total_batch % num_gpus
            nearest = [n for n in (floor, floor + num_gpus) if n >= num_gpus]
            raise RuntimeError(
                f"batch size {total_batch} does not divide evenly across {num_gpus} GPUs "
                f"({total_batch / num_gpus:.2f} images per GPU) — "
                f"try {' or '.join(str(n) for n in nearest)}"
            )
        batch_size = str(total_batch // num_gpus)
        console.print(
            f"[dim]{total_batch} total = {batch_size} per GPU x {num_gpus} — the schedule, lr "
            f"and eval readout all follow the total, so this matches a single-GPU run at "
            f"batch {total_batch}[/dim]"
        )
    else:
        batch_size = ask("Batch size", defaults.batch_size)
    lr = ask(f"Learning rate (at batch {defaults.reference_batch_size}, scaled from there)",
             lr_default)

    resume_default = latest_step_checkpoint(defaults.checkpoint_dir)
    if require_checkpoint and not resume_default:
        raise RuntimeError(
            "finetuning needs a pretrained checkpoint, and no vqgan_step*.pt was found in "
            f"{defaults.checkpoint_dir} — pretrain with 'Train' first"
        )
    if resume_default:
        # ask() returns the bracketed default on blank input (same as every
        # other prompt here), so once a checkpoint is found, blank means
        # "use it". "scratch" is the explicit escape hatch.
        resume = ask("Resume from checkpoint (blank = use this, or type 'scratch')", resume_default)
        if resume.strip().lower() == "scratch":
            resume = ""
    else:
        resume = ask("Resume from checkpoint (blank = train from scratch)", defaults.resume)

    use_prep = ask("Use a prep data file for the eval subset? (y/n)", "n")
    eval_prep_file = ""
    if use_prep.strip().lower().startswith("y"):
        eval_prep_file = ask("Prep data file", defaults.eval_prep_file or "data/eval_prep.pt")

    argv = [
        "--source", source,
        "--max-steps", max_steps, "--batch-size", batch_size, "--lr", lr,
    ]
    if resume:
        argv += ["--resume", resume]
    if eval_prep_file:
        argv += ["--eval-prep-file", eval_prep_file]
    if num_gpus > 1:
        _launch_ddp(argv, num_gpus)
    else:
        train_vqgan.main(argv)


def run_train_vqgan():
    """Pretrain on the downloaded dataset, using its own train/val/test split."""
    console.print("[dim]source: images-parquet — splits come from the dataset itself[/dim]")
    _run_training("parquet", lr_default=VQGANTrainConfig().lr, require_checkpoint=False)


def run_finetune_vqgan():
    """Continue training on the user's own images/ split."""
    console.print("[dim]source: images/ — splits come from the ratios you set in 'Build data index'[/dim]")
    _run_training("folder", lr_default=1e-5, require_checkpoint=True)


def run_train_refine():
    """Attach a RefinementHead to a chosen backbone, or carry on training one.

    Its own menu entry rather than more questions inside Train/Finetune: the
    entry point (attach to a backbone vs. continue a run) and the freeze mode
    are decisions plain VQ training never has to make, and everyone training
    without a head would have to answer them anyway.

    The head's shape (--refine-hidden-channels, --refine-num-blocks) and its
    finer schedule knobs (--refine-lr, --refine-warmup-steps) are left at
    RefineConfig's defaults here. As the module docstring says, call
    scripts/train_vqgan.py directly for full control over every flag.
    """
    defaults = VQGANTrainConfig()
    console.print("[dim]source: images-parquet — same corpus as 'Train' (call "
                  "scripts/train_vqgan.py with --source folder to refine on images/)[/dim]")

    # Which of the two starting points this is. They load different amounts of
    # the same file, and train_vqgan.py refuses both flags at once, so the
    # menu asks once and passes exactly one.
    entry = ask("Attach refine to a VQGAN checkpoint (a), or resume an existing refine run (r)?",
                "a")
    resume_flag, vqgan_ckpt_flag = [], []
    if entry.strip().lower().startswith("r"):
        resume_default = (latest_step_checkpoint(defaults.checkpoint_dir)
                          or str(Path(defaults.checkpoint_dir) / "vqgan_last.pt"))
        resume = ask("Resume from checkpoint", resume_default)
        resume_flag = ["--resume", resume]
        console.print("[dim]pick the same train stage this checkpoint was saved under — the "
                      "generator's optimizer state is grouped by it[/dim]")
    else:
        default_checkpoint = str(Path(defaults.checkpoint_dir) / "vqgan_last.pt")
        vqgan_checkpoint = ask("VQGAN backbone checkpoint to attach the head to",
                               default_checkpoint)
        vqgan_ckpt_flag = ["--vqgan-checkpoint", vqgan_checkpoint]

    stage = ask("Train stage — joint (everything) / refine_only / vq_only", "refine_only")
    if stage not in RefineConfig.STAGES:
        # Caught by the menu loop and printed. Letting it through to
        # train_vqgan.py instead would reach argparse's parser.error(), which
        # raises SystemExit and would take the whole menu down with it.
        raise ValueError(f"train stage must be one of {', '.join(RefineConfig.STAGES)}, "
                         f"got {stage!r}")
    max_steps = ask("Max steps", defaults.max_steps)
    batch_size = ask("Batch size", defaults.batch_size)
    # Lower than Train's default for the same reason Finetune's is: this
    # starts from a backbone that is already converged, not from noise.
    lr = ask(f"Learning rate (at batch {defaults.reference_batch_size}, scaled from there)", 1e-5)

    use_prep = ask("Use a prep data file for the eval subset? (y/n)", "n")
    prep_flag = []
    if use_prep.strip().lower().startswith("y"):
        prep_flag = ["--eval-prep-file",
                     ask("Prep data file", defaults.eval_prep_file or "data/eval_prep.pt")]

    argv = [
        "--source", "parquet",
        "--max-steps", max_steps, "--batch-size", batch_size, "--lr", lr,
        "--refine-enabled", "true", "--refine-train-stage", stage,
        *resume_flag, *vqgan_ckpt_flag, *prep_flag,
    ]
    train_vqgan.main(argv)


def run_pack_result():
    defaults = VQGANTrainConfig()
    out_dir = Path(defaults.out_dir)
    checkpoint_dir = Path(defaults.checkpoint_dir)

    result_dir = out_dir / f"result_{datetime.now():%Y%m%d_%H%M%S}"
    result_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]created[/green] {result_dir}")

    config_path = result_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(defaults), f, indent=2)
    console.print(f"[green]wrote[/green] {config_path}")

    recon_files = sorted(out_dir.glob("recon_step*.png")) if out_dir.is_dir() else []
    for p in recon_files:
        shutil.move(str(p), str(result_dir / p.name))
    if recon_files:
        console.print(f"[green]moved[/green] {len(recon_files)} recon image(s) to {result_dir}")
    else:
        console.print("[yellow]no recon_step*.png files found[/yellow]")

    latest_ckpt = latest_step_checkpoint(checkpoint_dir)
    if latest_ckpt:
        latest_ckpt = Path(latest_ckpt)
        shutil.move(str(latest_ckpt), str(result_dir / latest_ckpt.name))
        console.print(f"[green]moved[/green] {latest_ckpt.name} to {result_dir}")
    else:
        console.print("[yellow]no vqgan_step*.pt checkpoint found (vqgan_last.pt is left alone)[/yellow]")


def run_evaluate():
    defaults = VQGANTrainConfig()
    default_checkpoint = str(Path(defaults.checkpoint_dir) / "vqgan_last.pt")
    vqgan_checkpoint = ask("VQGAN checkpoint", default_checkpoint)
    source = ask("Source (parquet/folder/all)", defaults.source)
    split = ask("Split (validation/test)", "validation")
    console.print("[dim]the whole split is scored — this can take hours on ImageNet[/dim]")
    evaluate.main([
        "--vqgan-checkpoint", vqgan_checkpoint,
        "--source", source, "--split", split,
    ])


def run_test():
    """Score the dataset's own test split the way the model is actually used:
    whole images, tiled and reassembled. Source and split are fixed — a test
    set that can be pointed somewhere else is not a test set."""
    defaults = VQGANTrainConfig()
    default_checkpoint = str(Path(defaults.checkpoint_dir) / "vqgan_last.pt")
    vqgan_checkpoint = ask("VQGAN checkpoint", default_checkpoint)
    console.print("[dim]source: images-parquet, test split — images are tiled, run in "
                  "batches, stitched back, then scored at full size[/dim]")
    max_images = ask("Images to score (0 = the whole split)", 2048)
    overlap = ask("Tile overlap (px)", 64)
    test_whole_image.main([
        "--vqgan-checkpoint", vqgan_checkpoint,
        "--source", "parquet", "--split", "test",
        "--max-images", max_images,
        "--overlap", overlap,
    ])


def run_reconstruct():
    defaults = VQGANTrainConfig()
    image = ask("Image file or directory", DataConfig().folder_root)
    default_checkpoint = str(Path(defaults.checkpoint_dir) / "vqgan_last.pt")
    vqgan_checkpoint = ask("VQGAN checkpoint", default_checkpoint)
    overlap = ask("Tile overlap (px)", 64)
    reconstruct.main([
        "--image", image,
        "--vqgan-checkpoint", vqgan_checkpoint,
        "--overlap", overlap,
        "--side-by-side",
    ])


def run_visualize_model():
    defaults = VQGANTrainConfig()
    default_checkpoint = str(Path(defaults.checkpoint_dir) / "vqgan_last.pt")
    vqgan_checkpoint = ask("VQGAN checkpoint", default_checkpoint)
    visualize_model.main(["--vqgan-checkpoint", vqgan_checkpoint])


def main_menu():
    options = {
        "1": ("Build data index", run_build_index),
        "2": ("Count crops (how many tiles the index yields)", run_count_crops),
        "3": ("Create prep data file (cache the eval subset)", run_prep_eval_data),
        "4": ("Train ViT-VQGAN (images-parquet)", run_train_vqgan),
        "5": ("Finetune ViT-VQGAN (images/)", run_finetune_vqgan),
        "6": ("Pack Result", run_pack_result),
        "7": ("Evaluate crops (FID, codebook usage)", run_evaluate),
        "8": ("Test whole images (tile + stitch + score)", run_test),
        "9": ("Reconstruct image (tile + stitch)", run_reconstruct),
        "10": ("Visualize model (TensorBoard graph)", run_visualize_model),
        "11": ("Train Refinement Head (attach / continue, joint / refine_only / vq_only)",
               run_train_refine),
        "0": ("Exit", None),
    }
    while True:
        console.print("\n[bold]=== ViT-VQGAN pipeline ===[/bold]")
        for key, (label, _) in options.items():
            console.print(f"  {key}) {label}")
        choice = input("> ").strip()
        if choice == "0" or choice not in options:
            console.print("bye")
            return
        _, action = options[choice]
        try:
            action()
        except Exception as e:  # keep the menu alive after a failed stage
            console.print(f"[red]error:[/red] {e}")


if __name__ == "__main__":
    try:
        main_menu()
    except (EOFError, KeyboardInterrupt):
        console.print("\nbye")
