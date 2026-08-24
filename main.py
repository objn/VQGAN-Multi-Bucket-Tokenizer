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


def ask(prompt: str, default) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else str(default)


def run_preprocess():
    root = ask("Raw images root", "images")
    out = ask("Output dir", "data/processed")
    val_frac = ask("Validation fraction", "0.20")
    preprocess.main(["--root", root, "--out", out, "--val-frac", val_frac])


def run_train_vqgan():
    data_dir = ask("Preprocessed data dir", "data/processed")
    epochs = ask("Epochs", "100")
    batch_size = ask("Batch size", "14")
    resume = ask("Resume from checkpoint (blank = train from scratch)", "")
    argv = ["--data-dir", data_dir, "--epochs", epochs, "--batch-size", batch_size]
    if resume:
        argv += ["--resume", resume]
    train_vqgan.launch(argv)


def run_evaluate():
    data_dir = ask("Preprocessed data dir", "data/processed")
    vqgan_checkpoint = ask("VQGAN checkpoint", "checkpoints/vqgan_last.pt")
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
