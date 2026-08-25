"""Interactive menu entrypoint for the VQGAN pipeline.

    python main.py

Each menu item is a thin wrapper around the corresponding scripts/*.py CLI —
for full control over every flag, call those scripts directly instead
(e.g. `python scripts/train_vqgan.py --epochs 100 --batch-size 14`).

Autoregressive Transformer generation is out of scope for now — this project
is focused on getting VQGAN encode/decode reconstruction quality right first.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts import evaluate, preprocess, train_vqgan
from vqgan.config import PreprocessConfig, VQGANTrainConfig


def ask(prompt: str, default) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else str(default)


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
    resume = ask("Resume from checkpoint (blank = train from scratch)", defaults.resume)
    argv = ["--data-dir", data_dir, "--epochs", epochs, "--batch-size", batch_size]
    if resume:
        argv += ["--resume", resume]
    train_vqgan.main(argv)


def run_evaluate():
    defaults = VQGANTrainConfig()
    data_dir = ask("Preprocessed data dir", defaults.data_dir)
    default_checkpoint = str(Path(defaults.checkpoint_dir) / "vqgan_last.pt")
    vqgan_checkpoint = ask("VQGAN checkpoint", default_checkpoint)
    evaluate.main(["--data-dir", data_dir, "--vqgan-checkpoint", vqgan_checkpoint])


def main_menu():
    options = {
        "1": ("Preprocess data", run_preprocess),
        "2": ("Train VQGAN", run_train_vqgan),
        "3": ("Evaluate (FID, codebook usage)", run_evaluate),
        "0": ("Exit", None),
    }
    while True:
        print("\n=== VQGAN pipeline ===")
        for key, (label, _) in options.items():
            print(f"  {key}) {label}")
        choice = input("> ").strip()
        if choice == "0" or choice not in options:
            print("bye")
            return
        _, action = options[choice]
        try:
            action()
        except Exception as e:  # keep the menu alive after a failed stage
            print(f"error: {e}")


if __name__ == "__main__":
    try:
        main_menu()
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
