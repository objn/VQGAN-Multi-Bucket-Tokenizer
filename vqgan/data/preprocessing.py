"""Pure, unit-testable preprocessing functions used by scripts/preprocess.py.

Pipeline per image: verify integrity -> filter by resolution -> resize (long side
-> canvas_size, short side scaled proportionally) -> crop short side down to a
multiple of `downsample_factor` (center crop) -> place top-left on a black
canvas_size x canvas_size canvas. The bottom/right margin (if any) is masked
out of the VQGAN reconstruction loss (see vqgan/training/train_step.py).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class DiscoveredImage:
    path: Path
    account: str


def discover_images(root) -> list[DiscoveredImage]:
    """Find images under `root` and assign each an account label.

    If `root` contains subdirectories, each immediate subdirectory is treated as
    one account (files searched recursively within it), using the folder name
    as-is. Otherwise, images are expected flat in `root` with filenames like
    "{account}_{...}.jpg" and the account is the prefix before the first
    underscore.
    """
    root = Path(root)
    subdirs = [p for p in root.iterdir() if p.is_dir()]

    found: list[DiscoveredImage] = []
    if subdirs:
        for account_dir in subdirs:
            for path in sorted(account_dir.rglob("*")):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    found.append(DiscoveredImage(path=path, account=account_dir.name))
    else:
        for path in sorted(root.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                account = path.stem.split("_", 1)[0]
                found.append(DiscoveredImage(path=path, account=account))
    return found


def is_valid_image(path) -> bool:
    """Open + verify an image is not corrupt/truncated."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def passes_resolution_filter(im: Image.Image, min_short_side: int) -> bool:
    w, h = im.size
    return min(w, h) >= min_short_side


def resize_and_crop(
    im: Image.Image, canvas_size: int = 256, downsample_factor: int = 8
) -> tuple[np.ndarray, int, int]:
    """Resize long side to `canvas_size`, floor short side to a multiple of
    `downsample_factor`, center-crop, and place top-left on a black canvas.

    Returns (canvas[canvas_size, canvas_size, 3] uint8, final_h, final_w).
    """
    im = im.convert("RGB")
    w, h = im.size

    if w >= h:
        new_w = canvas_size
        new_h = max(1, round(h * canvas_size / w))
    else:
        new_h = canvas_size
        new_w = max(1, round(w * canvas_size / h))

    resized = im.resize((new_w, new_h), Image.LANCZOS)

    final_w = max(downsample_factor, (new_w // downsample_factor) * downsample_factor)
    final_h = max(downsample_factor, (new_h // downsample_factor) * downsample_factor)

    left = (new_w - final_w) // 2
    top = (new_h - final_h) // 2
    cropped = resized.crop((left, top, left + final_w, top + final_h))

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:final_h, :final_w, :] = np.asarray(cropped, dtype=np.uint8)

    return canvas, final_h, final_w
