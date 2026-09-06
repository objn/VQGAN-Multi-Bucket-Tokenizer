"""Evaluation: FID (real val crops vs. reconstructions), codebook usage %,
and a spot-check grid PNG.

Usage:
    python scripts/evaluate.py --vqgan-checkpoint checkpoints/vqgan_last.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid, save_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqgan.config import DataConfig, VQGANTrainConfig
from vqgan.data import build_shards, build_val_batch
from vqgan.display import console, tqdm
from vqgan.eval import compute_statistics, extract_features, fid_from_stats, get_feature_extractor
from vqgan.models import VQGAN


def parse_args(argv=None):
    data_defaults = DataConfig()
    train_defaults = VQGANTrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", default=data_defaults.index_path)
    parser.add_argument(
        "--vqgan-checkpoint", default=str(Path(train_defaults.checkpoint_dir) / "vqgan_last.pt")
    )
    parser.add_argument("--num-images", type=int, default=2048, help="validation crops to score")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", dest="out_dir", default="outputs/eval")
    return parser.parse_args(argv)


def load_vqgan(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_config" not in ckpt:
        raise ValueError(
            f"{checkpoint_path} has no model_config entry — it predates the ViT-VQGAN rewrite "
            f"and holds CNN encoder/decoder weights that cannot be loaded."
        )
    vqgan = VQGAN(**ckpt["model_config"]).to(device)
    vqgan.load_state_dict(ckpt["vqgan"])
    vqgan.eval()
    return vqgan, ckpt["model_config"]


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqgan, model_config = load_vqgan(args.vqgan_checkpoint, device)

    with open(args.index_path, encoding="utf-8") as f:
        index = json.load(f)
    val_images = build_val_batch(
        build_shards(index, "validation"),
        tile_size=model_config["image_size"],
        num_images=args.num_images,
    )
    console.print(f"scoring {val_images.shape[0]} validation crops")

    feature_extractor = get_feature_extractor(device)
    vqgan.quantizer.reset_usage_stats()

    real_feats, recon_feats = [], []
    first_batch = None
    with torch.no_grad():
        for i in tqdm(range(0, val_images.shape[0], args.batch_size), desc="evaluating"):
            images = val_images[i:i + args.batch_size].to(device)
            recon = vqgan(images).recon
            if first_batch is None:
                first_batch = (images[:8].cpu(), recon[:8].cpu())
            real_feats.append(extract_features(feature_extractor, images))
            recon_feats.append(extract_features(feature_extractor, recon))

    real_feats = np.concatenate(real_feats, axis=0)
    recon_feats = np.concatenate(recon_feats, axis=0)
    mu_r, sigma_r = compute_statistics(real_feats)
    mu_f, sigma_f = compute_statistics(recon_feats)
    fid = fid_from_stats(mu_r, sigma_r, mu_f, sigma_f)

    usage_pct = vqgan.quantizer.codebook_usage_pct()
    console.print(f"[bold]FID[/bold] (real vs. recon): {fid:.3f}")
    console.print(f"[bold]codebook usage[/bold]: {usage_pct:.1f}%")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real, recon = first_batch
    grid = make_grid(torch.cat([real, recon], dim=0), nrow=real.shape[0])
    save_image((grid + 1) / 2, out_dir / "vqgan_spotcheck.png")
    console.print(f"[green]saved[/green] {out_dir / 'vqgan_spotcheck.png'}")


if __name__ == "__main__":
    main()
