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
    defaults = DataConfig()
    folder_root = ask("Your own images dir", defaults.folder_root)
    manifest_dir = ask("Parquet manifest dir", defaults.manifest_dir)
    out = ask("Index output", defaults.index_path)
    build_index.main(["--folder-root", folder_root, "--manifest-dir", manifest_dir, "--out", out])


def run_train_vqgan():
    defaults = VQGANTrainConfig()
    index_path = ask("Data index", defaults.index_path)
    max_steps = ask("Max steps", defaults.max_steps)
    batch_size = ask("Batch size", defaults.batch_size)

    resume_default = latest_step_checkpoint(defaults.checkpoint_dir)
    if resume_default:
        # ask() returns the bracketed default on blank input (same as every
        # other prompt here), so once a checkpoint is found, blank means
        # "use it". "scratch" is the explicit escape hatch.
        resume = ask("Resume from checkpoint (blank = use this, or type 'scratch')", resume_default)
        if resume.strip().lower() == "scratch":
            resume = ""
    else:
        resume = ask("Resume from checkpoint (blank = train from scratch)", defaults.resume)

    argv = ["--index-path", index_path, "--max-steps", max_steps, "--batch-size", batch_size]
    if resume:
        argv += ["--resume", resume]
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
    index_path = ask("Data index", DataConfig().index_path)
    default_checkpoint = str(Path(defaults.checkpoint_dir) / "vqgan_last.pt")
    vqgan_checkpoint = ask("VQGAN checkpoint", default_checkpoint)
    evaluate.main(["--index-path", index_path, "--vqgan-checkpoint", vqgan_checkpoint])


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
        "2": ("Train ViT-VQGAN", run_train_vqgan),
        "3": ("Pack Result", run_pack_result),
        "4": ("Evaluate (FID, codebook usage)", run_evaluate),
        "5": ("Reconstruct image (tile + stitch)", run_reconstruct),
        "6": ("Visualize model (TensorBoard graph)", run_visualize_model),
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
