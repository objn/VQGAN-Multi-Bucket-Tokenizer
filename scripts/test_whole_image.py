"""Test stage: score whole images, not crops.

Everything else in this pipeline measures the model on 256px crops, because
that is the only thing it ever sees. This measures what the user actually gets:
each test image is cut into overlapping tiles, the tiles are pushed through the
model in batches, and the pieces are blended back into an image of exactly the
original size — then compared, whole, against the original. Seams between
independently-decoded tiles are invisible to a per-crop metric and obvious in
this one.

Images smaller than one tile are counted and skipped; they cannot be tiled at
native resolution and the model never upscales.

Usage:
    python scripts/test_whole_image.py
    python scripts/test_whole_image.py --max-images 0        # the whole split
    python scripts/test_whole_image.py --split validation --source folder
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqgan.config import DataConfig, VQGANTrainConfig
from vqgan.data import WholeImageDataset, build_shards, collate_single
from vqgan.display import console, tqdm
from vqgan.eval import (
    compute_statistics,
    extract_features,
    fid_from_stats,
    get_feature_extractor,
    psnr,
    reconstruct_tiled,
)
from vqgan.losses import LPIPS_AVAILABLE, get_lpips_model
from vqgan.models import VQGAN

# Inception's pool features are 2048-d, so a covariance estimated from fewer
# than that many images is rank-deficient and its FID is biased low. Runs are
# still comparable to each other at the same count; across counts they are not.
FID_MIN_SAMPLES = 2048


def parse_args(argv=None):
    data_defaults = DataConfig()
    train_defaults = VQGANTrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", default=data_defaults.index_path)
    parser.add_argument(
        "--vqgan-checkpoint", default=str(Path(train_defaults.checkpoint_dir) / "vqgan_last.pt")
    )
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument(
        "--source", default="parquet", choices=("parquet", "folder", "all"),
        help="parquet = the downloaded dataset's own split; folder = images/",
    )
    parser.add_argument(
        "--max-images", type=int, default=FID_MIN_SAMPLES,
        help="stop after this many scored images; 0 = the whole split",
    )
    parser.add_argument(
        "--overlap", type=int, default=64,
        help="pixels of overlap between neighbouring tiles, cross-faded on stitch",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="tiles decoded at once")
    parser.add_argument("--num-workers", type=int, default=train_defaults.num_workers)
    parser.add_argument("--no-lpips", dest="lpips", action="store_false")
    parser.add_argument("--save-samples", type=int, default=4, help="original|recon pairs to write")
    parser.add_argument("--out", dest="out_dir", default="outputs/test")
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


@torch.no_grad()
def lpips_whole(model, recon, original, device) -> float:
    """LPIPS over the entire image.

    A 3000x3000 photo through VGG does not fit next to the VQGAN on a 12GB
    card, and there is no smaller-but-still-honest version of a whole-image
    metric, so the fallback is the CPU: slow, rare, and correct.
    """
    pair = (recon[None], original[None])
    try:
        return float(model(pair[0].to(device), pair[1].to(device)).mean())
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float(model.cpu()(*pair).mean())
    finally:
        model.to(device)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqgan, model_config = load_vqgan(args.vqgan_checkpoint, device)
    tile_size = model_config["image_size"]
    if not 0 <= args.overlap < tile_size:
        raise ValueError(f"--overlap must be in [0, {tile_size})")

    with open(args.index_path, encoding="utf-8") as f:
        index = json.load(f)
    shards = build_shards(index, args.split, args.source)
    if not shards:
        raise RuntimeError(f"no {args.split} shards for source={args.source!r} in {args.index_path}")

    loader = DataLoader(
        WholeImageDataset(shards, min_size=tile_size),
        batch_size=None,
        collate_fn=collate_single,
        num_workers=args.num_workers,
    )
    limit = args.max_images if args.max_images > 0 else None
    console.print(
        f"whole-image test on the {args.split} split from {args.source} "
        f"({len(shards)} shard(s)), tile {tile_size}px, overlap {args.overlap}px"
    )
    console.print(f"[dim]{'all images' if limit is None else f'first {limit:,} images'}, "
                  f"reassembled and scored at full size[/dim]")

    feature_extractor = get_feature_extractor(device)
    lpips_model = get_lpips_model(device) if (args.lpips and LPIPS_AVAILABLE) else None
    if args.lpips and not LPIPS_AVAILABLE:
        console.print("[yellow]lpips not installed — skipping the LPIPS column[/yellow]")
    vqgan.quantizer.reset_usage_stats()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    l1_scores, psnr_scores, lpips_scores, pixel_counts = [], [], [], []
    real_feats, recon_feats = [], []
    n_scored = n_small = 0

    progress = tqdm(loader, total=limit, desc="images")
    for source in progress:
        if source is None:
            n_small += 1
            continue

        recon = reconstruct_tiled(
            vqgan, source, tile_size, args.overlap, device, args.batch_size
        )

        l1_scores.append(float((recon - source).abs().mean()))
        psnr_scores.append(psnr(recon, source))
        pixel_counts.append(source.shape[1] * source.shape[2])
        if lpips_model is not None:
            lpips_scores.append(lpips_whole(lpips_model, recon, source, device))
        real_feats.append(extract_features(feature_extractor, source[None].to(device)))
        recon_feats.append(extract_features(feature_extractor, recon[None].to(device)))

        if n_scored < args.save_samples:
            pair = torch.cat([source, recon], dim=2)
            save_image((pair + 1) / 2, out_dir / f"test_pair_{n_scored:02d}.png")

        n_scored += 1
        progress.set_postfix(L1=f"{np.mean(l1_scores):.4f}", PSNR=f"{np.mean(psnr_scores):.2f}dB")
        if limit is not None and n_scored >= limit:
            break
    progress.close()

    if not n_scored:
        raise RuntimeError(f"no image in the {args.split} split was large enough to tile")

    results = {
        "checkpoint": args.vqgan_checkpoint,
        "split": args.split,
        "source": args.source,
        "tile_size": tile_size,
        "overlap": args.overlap,
        "images_scored": n_scored,
        "images_skipped_too_small": n_small,
        "megapixels_scored": round(sum(pixel_counts) / 1e6, 2),
        "l1": float(np.mean(l1_scores)),
        "psnr_db": float(np.mean(psnr_scores)),
        "codebook_usage_pct": float(vqgan.quantizer.codebook_usage_pct()),
    }
    if lpips_scores:
        results["lpips"] = float(np.mean(lpips_scores))

    mu_r, sigma_r = compute_statistics(np.concatenate(real_feats, axis=0))
    mu_f, sigma_f = compute_statistics(np.concatenate(recon_feats, axis=0))
    results["fid"] = fid_from_stats(mu_r, sigma_r, mu_f, sigma_f)

    console.print(
        f"scored {n_scored:,} whole image(s) "
        f"({results['megapixels_scored']:,.1f} MP), skipped {n_small:,} under {tile_size}px"
    )
    console.print(f"[bold]L1[/bold] (whole image): {results['l1']:.4f}")
    console.print(f"[bold]PSNR[/bold]: {results['psnr_db']:.2f} dB")
    if "lpips" in results:
        console.print(f"[bold]LPIPS[/bold]: {results['lpips']:.4f}")
    console.print(f"[bold]FID[/bold] (real vs. recon): {results['fid']:.3f}")
    console.print(f"[bold]codebook usage[/bold]: {results['codebook_usage_pct']:.1f}%")
    if n_scored < FID_MIN_SAMPLES:
        console.print(
            f"[yellow]FID is biased at {n_scored:,} images[/yellow] — comparable to other runs "
            f"scored at the same count, not to one at {FID_MIN_SAMPLES:,}+"
        )

    results_path = out_dir / "test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]wrote[/green] {results_path}")
    if args.save_samples:
        console.print(f"[green]saved[/green] {min(args.save_samples, n_scored)} original|recon pair(s) to {out_dir}")


if __name__ == "__main__":
    main()
