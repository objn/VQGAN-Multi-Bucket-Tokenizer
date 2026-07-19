"""Bucketed image-folder dataset -- no captions/metadata required for VQGAN training.

Each image is assigned to its nearest aspect-ratio bucket (see buckets.py) once at
dataset construction time, then resized (preserving aspect ratio, "cover" fit) and
cropped to that bucket's exact resolution on every access.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from vqgan.data.buckets import Bucket, DEFAULT_BUCKETS, assign_bucket

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# small resize margin beyond exact "cover" fit so RandomCrop has room to jitter
TRAIN_CROP_MARGIN = 1.05


def _resize_to_cover(image: Image.Image, target_width: int, target_height: int, margin: float = 1.0) -> Image.Image:
    """Resizes preserving aspect ratio so the image is at least (target_w, target_h) in
    both dimensions -- "cover" fit, as opposed to naive stretch/squash."""
    orig_w, orig_h = image.size
    scale = max(target_width / orig_w, target_height / orig_h) * margin
    new_w = max(target_width, round(orig_w * scale))
    new_h = max(target_height, round(orig_h * scale))
    return image.resize((new_w, new_h), Image.BICUBIC)


class BucketedImageDataset(Dataset):
    """Recursively collects images under `root`, assigns each to its nearest bucket by
    aspect ratio, and returns tensors normalized to [-1, 1] at that bucket's resolution.
    """

    def __init__(
        self,
        root: str,
        buckets: tuple[Bucket, ...] = DEFAULT_BUCKETS,
        horizontal_flip: bool = True,
        train: bool = True,
    ):
        self.root = Path(root)
        self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")

        self.buckets = list(buckets)
        self.horizontal_flip = horizontal_flip
        self.train = train

        # header-only reads (PIL doesn't decode pixel data for .size), so this is cheap
        # even over 100k+ images
        self.bucket_ids: list[int] = []
        for path in self.paths:
            with Image.open(path) as img:
                width, height = img.size
            bucket = assign_bucket(width, height, self.buckets)
            self.bucket_ids.append(self.buckets.index(bucket))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Returns (image, bucket_id). Every item in a batch sampled by
        BucketedBatchSampler shares one bucket_id, so callers can read it off
        batch_bucket_ids[0] to know which bucket the whole batch belongs to."""
        path = self.paths[idx]
        bucket_id = self.bucket_ids[idx]
        bucket = self.buckets[bucket_id]
        image = Image.open(path).convert("RGB")

        if self.train:
            image = _resize_to_cover(image, bucket.width, bucket.height, margin=TRAIN_CROP_MARGIN)
            crop = transforms.RandomCrop((bucket.height, bucket.width))
        else:
            image = _resize_to_cover(image, bucket.width, bucket.height)
            crop = transforms.CenterCrop((bucket.height, bucket.width))

        tf = [crop]
        if self.train and self.horizontal_flip:
            tf.append(transforms.RandomHorizontalFlip())
        tf += [
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
        ]
        return transforms.Compose(tf)(image), bucket_id
