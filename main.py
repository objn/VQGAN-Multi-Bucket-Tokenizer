"""Interactive menu entrypoint for the ViT-VQGAN pipeline.

    python main.py

Each menu item is a thin wrapper around the corresponding scripts/*.py CLI —
for full control over every flag, call those scripts directly instead
(e.g. `python scripts/train_vqgan.py --max-steps 200000 --batch-size 8`).

Autoregressive Transformer generation is out of scope for now — this project
is focused on getting ViT-VQGAN encode/decode reconstruction quality right first.
"""

import json
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts import build_index, evaluate, reconstruct, train_vqgan, visualize_model
from vqgan.config import DataConfig, VQGANTrainConfig
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


def _run_training(source: str, *, lr_default, require_checkpoint: bool):
    defaults = VQGANTrainConfig()
    # No prompt for the index path: it is a fixed project location, and typing
    # a different one here without matching it in Build data index would
    # silently train on a stale index.
    max_steps = ask("Max steps", defaults.max_steps)
    batch_size = ask("Batch size", defaults.batch_size)
    lr = ask("Learning rate", lr_default)

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

    argv = [
        "--source", source,
        "--max-steps", max_steps, "--batch-size", batch_size, "--lr", lr,
    ]
    if resume:
        argv += ["--resume", resume]
    train_vqgan.main(argv)


def run_train_vqgan():
    """Pretrain on the downloaded dataset, using its own train/val/test split."""
    console.print("[dim]source: images-parquet — splits come from the dataset itself[/dim]")
    _run_training("parquet", lr_default=VQGANTrainConfig().lr, require_checkpoint=False)


def run_finetune_vqgan():
    """Continue training on the user's own images/ split."""
    console.print("[dim]source: images/ — splits come from the ratios you set in 'Build data index'[/dim]")
    _run_training("folder", lr_default=1e-5, require_checkpoint=True)


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
        "2": ("Train ViT-VQGAN (images-parquet)", run_train_vqgan),
        "3": ("Finetune ViT-VQGAN (images/)", run_finetune_vqgan),
        "4": ("Pack Result", run_pack_result),
        "5": ("Evaluate (FID, codebook usage)", run_evaluate),
        "6": ("Reconstruct image (tile + stitch)", run_reconstruct),
        "7": ("Visualize model (TensorBoard graph)", run_visualize_model),
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
