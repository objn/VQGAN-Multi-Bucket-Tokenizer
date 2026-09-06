"""Encode and decode a full-resolution image by tiles, then stitch it back.

The model only ever sees tile_size x tile_size crops, so a 1400x1800 photo is
cut into overlapping tiles, each one round-tripped through the ViT-VQGAN, and
the results blended back into an image of exactly the original size. Tiles at
the right and bottom edges are shifted inward rather than padded, and the
overlap is cross-faded, so there are no seams and no black margin.

Usage:
    python scripts/reconstruct.py --image images/kaemissbabe/photo.jpg
    python scripts/reconstruct.py --image images/kaemissbabe --overlap 64
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqgan.config import VQGANTrainConfig
from vqgan.data.sources import IMAGE_EXTENSIONS
from vqgan.data.tiling import plan_tiles, stitch_tiles
from vqgan.display import console, tqdm
from vqgan.models import VQGAN


def parse_args(argv=None):
    defaults = VQGANTrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="image file or a directory of them")
    parser.add_argument(
        "--vqgan-checkpoint", default=str(Path(defaults.checkpoint_dir) / "vqgan_last.pt")
    )
    parser.add_argument(
        "--overlap", type=int, default=64,
        help="pixels of overlap between neighbouring tiles, cross-faded on stitch",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="tiles decoded at once")
    parser.add_argument("--out", dest="out_dir", default="outputs/reconstruct")
    parser.add_argument(
        "--side-by-side", action="store_true", help="also write an original|reconstruction pair"
    )
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
def reconstruct_image(vqgan, image, tile_size, overlap, device, batch_size):
    """PIL image -> [3, H, W] reconstruction in [-1, 1] at the original size."""
    pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w = pixels.shape[:2]
    if h < tile_size or w < tile_size:
        raise ValueError(f"image is {w}x{h}, smaller than the {tile_size}px tile")

    source = torch.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1.0
    origins = plan_tiles(h, w, tile_size, overlap)

    recon_tiles = []
    for i in tqdm(range(0, len(origins), batch_size), desc="tiles", leave=False):
        chunk = origins[i:i + batch_size]
        batch = torch.stack(
            [source[:, top:top + tile_size, left:left + tile_size] for top, left in chunk]
        ).to(device)
        recon_tiles.append(vqgan(batch).recon.cpu())

    return stitch_tiles(torch.cat(recon_tiles), origins, (h, w), overlap)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqgan, model_config = load_vqgan(args.vqgan_checkpoint, device)
    tile_size = model_config["image_size"]
    if not 0 <= args.overlap < tile_size:
        raise ValueError(f"--overlap must be in [0, {tile_size})")
    console.print(f"tile {tile_size}px, overlap {args.overlap}px")

    source = Path(args.image)
    if source.is_dir():
        paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        paths = [source]
    if not paths:
        console.print(f"[yellow]no images found at {source}[/yellow]")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            recon = reconstruct_image(
                vqgan, image, tile_size, args.overlap, device, args.batch_size
            )
            original = torch.from_numpy(
                np.array(image, dtype=np.uint8)
            ).permute(2, 0, 1).float() / 127.5 - 1.0

        out_path = out_dir / f"{path.stem}_recon.png"
        save_image((recon + 1) / 2, out_path)
        console.print(f"[green]saved[/green] {out_path}  ({recon.shape[2]}x{recon.shape[1]})")

        if args.side_by_side:
            pair_path = out_dir / f"{path.stem}_pair.png"
            save_image((torch.cat([original, recon], dim=2) + 1) / 2, pair_path)
            console.print(f"[green]saved[/green] {pair_path}")


if __name__ == "__main__":
    main()
