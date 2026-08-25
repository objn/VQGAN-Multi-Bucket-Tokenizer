"""Interactive menu entrypoint for the VQGAN pipeline.

    python main.py

Each menu item is a thin wrapper around the corresponding scripts/*.py CLI —
for full control over every flag, call those scripts directly instead
(e.g. `python scripts/train_vqgan.py --epochs 100 --batch-size 14`).

Autoregressive Transformer generation is out of scope for now — this project
is focused on getting VQGAN encode/decode reconstruction quality right first.
"""

import json
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts import evaluate, preprocess, train_vqgan, visualize_model
from vqgan.config import PreprocessConfig, VQGANTrainConfig
from vqgan.display import console

_EPOCH_CKPT_RE = re.compile(r"vqgan_epoch(\d+)\.pt")


def ask(prompt: str, default) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else str(default)


def latest_epoch_checkpoint(checkpoint_dir) -> str:
    """Newest vqgan_epochXXXX.pt in checkpoint_dir by epoch number — never
    vqgan_last.pt, which gets overwritten every run and isn't tied to a
    specific epoch. Returns "" if none exist."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return ""
    candidates = []
    for p in checkpoint_dir.iterdir():
        m = _EPOCH_CKPT_RE.fullmatch(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return ""
    return str(max(candidates, key=lambda t: t[0])[1])


def run_preprocess():
    defaults = PreprocessConfig()
    root = ask("Raw images root", defaults.root)
    out = ask("Output dir", defaults.out_dir)
    val_frac = ask("Validation fraction", defaults.val_frac)
    preprocess.main(["--root", root, "--out", out, "--val-frac", val_frac])


def run_train_vqgan():
    defaults = VQGANTrainConfig()
    data_dir = ask("Preprocessed data dir", defaults.data_dir)
    epochs = ask("Epochs", defaults.epochs)
    batch_size = ask("Batch size", defaults.batch_size)

    resume_default = latest_epoch_checkpoint(defaults.checkpoint_dir)
    if resume_default:
        # ask() returns the bracketed default on blank input (same as every
        # other prompt here), so once a checkpoint is found, blank now means
        # "use it" — not "train from scratch" like the old hint text said.
        # "scratch" is the explicit escape hatch for the from-scratch case.
        resume = ask("Resume from checkpoint (blank = use this, or type 'scratch')", resume_default)
        if resume.strip().lower() == "scratch":
            resume = ""
    else:
        resume = ask("Resume from checkpoint (blank = train from scratch)", defaults.resume)

    argv = ["--data-dir", data_dir, "--epochs", epochs, "--batch-size", batch_size]
    if resume:
        argv += ["--resume", resume]
    train_vqgan.main(argv)


def run_finetune_vqgan():
    console.print("[yellow]FineTune VQGAN: not implemented yet[/yellow]")


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

    recon_files = sorted(out_dir.glob("recon_epoch*.png")) if out_dir.is_dir() else []
    for p in recon_files:
        shutil.move(str(p), str(result_dir / p.name))
    if recon_files:
        console.print(f"[green]moved[/green] {len(recon_files)} recon image(s) to {result_dir}")
    else:
        console.print("[yellow]no recon_epoch*.png files found[/yellow]")

    latest_ckpt = latest_epoch_checkpoint(checkpoint_dir)
    if latest_ckpt:
        latest_ckpt = Path(latest_ckpt)
        shutil.move(str(latest_ckpt), str(result_dir / latest_ckpt.name))
        console.print(f"[green]moved[/green] {latest_ckpt.name} to {result_dir}")
    else:
        console.print("[yellow]no vqgan_epoch*.pt checkpoint found (vqgan_last.pt is left alone)[/yellow]")


def run_evaluate():
    defaults = VQGANTrainConfig()
    data_dir = ask("Preprocessed data dir", defaults.data_dir)
    default_checkpoint = str(Path(defaults.checkpoint_dir) / "vqgan_last.pt")
    vqgan_checkpoint = ask("VQGAN checkpoint", default_checkpoint)
    evaluate.main(["--data-dir", data_dir, "--vqgan-checkpoint", vqgan_checkpoint])


def run_visualize_model():
    defaults = VQGANTrainConfig()
    default_checkpoint = str(Path(defaults.checkpoint_dir) / "vqgan_last.pt")
    vqgan_checkpoint = ask("VQGAN checkpoint", default_checkpoint)
    canvas_size = ask("Canvas size (dummy input for tracing)", PreprocessConfig().canvas_size)
    visualize_model.main(["--vqgan-checkpoint", vqgan_checkpoint, "--canvas-size", canvas_size])


def main_menu():
    options = {
        "1": ("Preprocess data", run_preprocess),
        "2": ("Train VQGAN", run_train_vqgan),
        "3": ("FineTune VQGAN", run_finetune_vqgan),
        "4": ("Pack Result", run_pack_result),
        "5": ("Evaluate (FID, codebook usage)", run_evaluate),
        "6": ("Visualize model (TensorBoard graph)", run_visualize_model),
        "0": ("Exit", None),
    }
    while True:
        console.print("\n[bold]=== VQGAN pipeline ===[/bold]")
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
