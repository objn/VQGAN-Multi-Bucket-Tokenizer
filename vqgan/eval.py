"""Standalone evaluation: rFID, LPIPS, PSNR, and codebook utilization on a held-out val split.

Computes both overall and per-bucket metrics -- a bucket with less training data (e.g.
extreme ratios like 16:9) may lag behind the more common ones, and that needs to be
visible rather than averaged away.

Usage:
    uv run python -m vqgan.eval --checkpoint runs/vqgan-multi/checkpoints/latest.pt --val-dir data/val
"""
import argparse
from collections import defaultdict

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3

from vqgan.config import ModelConfig
from vqgan.data.buckets import Bucket, DEFAULT_BUCKETS
from vqgan.data.dataset import BucketedImageDataset
from vqgan.models.vqgan import VQGAN

MIN_ACCEPTABLE_CODEBOOK_UTILIZATION = 0.5  # below this, warn that codebook capacity is being wasted
MIN_IMAGES_FOR_BUCKET_FID = 20  # Frechet distance needs enough samples for a stable covariance estimate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VQGAN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches-per-bucket", type=int, default=None, help="limit eval to N batches per bucket for a quick check")
    return parser.parse_args()


class InceptionFeatureExtractor(torch.nn.Module):
    """Pool3 (2048-d) features from torchvision's Inception v3, for FID."""

    def __init__(self):
        super().__init__()
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = torch.nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # inception expects 299x299, ImageNet-normalized, input in [0, 1]
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.net(x)


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def psnr(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """x, x_recon in [-1, 1]. Converts to [0, 1] before computing PSNR."""
    x01 = (x.clamp(-1, 1) + 1) / 2
    r01 = (x_recon.clamp(-1, 1) + 1) / 2
    mse = F.mse_loss(r01, x01, reduction="none").mean(dim=[1, 2, 3])
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


@torch.no_grad()
def evaluate(checkpoint_path: str, val_dir: str, batch_size: int, num_workers: int,
             max_batches_per_bucket: int | None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**ckpt["config"]["model"])
    bucket_dicts = ckpt["config"].get("data", {}).get("buckets")
    buckets = [Bucket(**b) for b in bucket_dicts] if bucket_dicts else list(DEFAULT_BUCKETS)

    model = VQGAN(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = BucketedImageDataset(val_dir, buckets=buckets, train=False)
    # group val indices by bucket and run one single-bucket DataLoader per bucket, so
    # every batch is naturally homogeneous without needing the training batch sampler
    indices_by_bucket: dict[int, list] = defaultdict(list)
    for idx, bucket_id in enumerate(dataset.bucket_ids):
        indices_by_bucket[bucket_id].append(idx)

    lpips_net = lpips.LPIPS(net="vgg").to(device).eval()
    inception = InceptionFeatureExtractor().to(device).eval()

    overall_lpips, overall_psnr = [], []
    overall_real_feats, overall_fake_feats = [], []
    code_counts = torch.zeros(model_config.codebook_size, dtype=torch.long)

    per_bucket: dict[str, dict] = {}

    for bucket_id, indices in indices_by_bucket.items():
        bucket = buckets[bucket_id]
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        bucket_lpips, bucket_psnr = [], []
        bucket_real_feats, bucket_fake_feats = [], []

        for batch_idx, (images, _) in enumerate(loader):
            if max_batches_per_bucket is not None and batch_idx >= max_batches_per_bucket:
                break
            images = images.to(device)
            out = model(images)
            x_recon = out["reconstruction"]

            bucket_lpips.append(lpips_net(images, x_recon).view(-1).cpu())
            bucket_psnr.append(psnr(images, x_recon).cpu())

            real01 = (images.clamp(-1, 1) + 1) / 2
            fake01 = (x_recon.clamp(-1, 1) + 1) / 2
            bucket_real_feats.append(inception(real01).cpu())
            bucket_fake_feats.append(inception(fake01).cpu())

            code_counts += torch.bincount(out["indices"].reshape(-1).cpu(), minlength=model_config.codebook_size)

        if not bucket_lpips:
            continue

        bucket_real = torch.cat(bucket_real_feats).numpy()
        bucket_fake = torch.cat(bucket_fake_feats).numpy()

        bucket_result = {
            "num_images": len(bucket_real),
            "LPIPS": torch.cat(bucket_lpips).mean().item(),
            "PSNR": torch.cat(bucket_psnr).mean().item(),
        }
        if len(bucket_real) >= MIN_IMAGES_FOR_BUCKET_FID:
            mu_r, sigma_r = bucket_real.mean(axis=0), np.cov(bucket_real, rowvar=False)
            mu_f, sigma_f = bucket_fake.mean(axis=0), np.cov(bucket_fake, rowvar=False)
            bucket_result["rFID"] = frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
        else:
            bucket_result["rFID"] = None  # too few images in this bucket for a stable FID estimate

        per_bucket[bucket.name] = bucket_result

        overall_lpips.append(torch.cat(bucket_lpips))
        overall_psnr.append(torch.cat(bucket_psnr))
        overall_real_feats.append(bucket_real)
        overall_fake_feats.append(bucket_fake)

    overall_real = np.concatenate(overall_real_feats, axis=0)
    overall_fake = np.concatenate(overall_fake_feats, axis=0)
    mu_real, sigma_real = overall_real.mean(axis=0), np.cov(overall_real, rowvar=False)
    mu_fake, sigma_fake = overall_fake.mean(axis=0), np.cov(overall_fake, rowvar=False)
    overall_rfid = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    codebook_utilization = (code_counts > 0).float().mean().item()

    return {
        "rFID": overall_rfid,
        "LPIPS": torch.cat(overall_lpips).mean().item(),
        "PSNR": torch.cat(overall_psnr).mean().item(),
        "codebook_utilization": codebook_utilization,
        "num_val_images": len(overall_real),
        "per_bucket": per_bucket,
    }


def main() -> None:
    args = parse_args()
    results = evaluate(
        args.checkpoint, args.val_dir, args.batch_size, args.num_workers, args.max_batches_per_bucket
    )

    print("\n=== VQGAN Evaluation (overall) ===")
    print(f"Images evaluated:      {results['num_val_images']}")
    print(f"rFID (lower better):   {results['rFID']:.4f}  (target < 2.0)")
    print(f"LPIPS (lower better):  {results['LPIPS']:.4f}")
    print(f"PSNR (higher better):  {results['PSNR']:.2f} dB")
    print(f"Codebook utilization:  {results['codebook_utilization'] * 100:.1f}%")

    print("\n=== Per-bucket breakdown ===")
    header = f"{'bucket':<8} {'images':>8} {'rFID':>10} {'LPIPS':>10} {'PSNR':>10}"
    print(header)
    for name, b in sorted(results["per_bucket"].items()):
        rfid_str = f"{b['rFID']:.4f}" if b["rFID"] is not None else "n/a (few)"
        print(f"{name:<8} {b['num_images']:>8} {rfid_str:>10} {b['LPIPS']:>10.4f} {b['PSNR']:>10.2f}")

    if results["codebook_utilization"] < MIN_ACCEPTABLE_CODEBOOK_UTILIZATION:
        print(
            f"\nWARNING: codebook utilization is below "
            f"{MIN_ACCEPTABLE_CODEBOOK_UTILIZATION * 100:.0f}% -- a large fraction of codebook "
            f"capacity is unused. Consider EMA updates, codebook resets for dead codes, or a "
            f"smaller codebook size."
        )


if __name__ == "__main__":
    main()
