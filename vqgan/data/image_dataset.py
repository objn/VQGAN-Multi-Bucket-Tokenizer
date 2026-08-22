import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class PixelDataset(Dataset):
    """Reads the memmap'd preprocessed canvas array + metadata produced by
    scripts/preprocess.py. Returns normalized [-1, 1] pixels, a valid-region
    mask (1 where real content is, 0 in the black pad margin), and the index.

    Augmentation (train split only) is confined to each image's valid
    sub-rectangle so the top-left content anchoring survives: flipping or
    cropping the full padded canvas would move content into (or out of) the
    pad margin and break the metadata-driven token padding downstream.
    """

    def __init__(self, data_dir, split: str = "train", augment: bool | None = None):
        data_dir = Path(data_dir)
        self.pixels = np.load(data_dir / "pixels.npy", mmap_mode="r")

        records = {}
        with open(data_dir / "metadata.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["index"]] = rec
        self.records = records

        with open(data_dir / "splits.json", encoding="utf-8") as f:
            splits = json.load(f)
        self.indices = splits[split]

        self.augment = augment if augment is not None else (split == "train")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        rec = self.records[idx]
        final_h, final_w = rec["final_h"], rec["final_w"]

        canvas = np.array(self.pixels[idx])  # copy out of the memmap, HWC uint8
        img = torch.from_numpy(canvas).permute(2, 0, 1)  # CHW uint8
        content = img[:, :final_h, :final_w]

        if self.augment:
            content = self._augment(content, final_h, final_w)

        out = torch.zeros_like(img)
        out[:, :final_h, :final_w] = content

        pixels = out.float() / 127.5 - 1.0
        valid_mask = torch.zeros(1, img.shape[1], img.shape[2], dtype=torch.float32)
        valid_mask[:, :final_h, :final_w] = 1.0

        return pixels, valid_mask, idx

    def _augment(self, content: torch.Tensor, final_h: int, final_w: int) -> torch.Tensor:
        if random.random() < 0.5:
            content = TF.hflip(content)

        jitter_h = max(1, int(final_h * 0.1))
        jitter_w = max(1, int(final_w * 0.1))
        top = random.randint(0, jitter_h)
        left = random.randint(0, jitter_w)
        bottom = final_h - random.randint(0, jitter_h)
        right = final_w - random.randint(0, jitter_w)
        if bottom - top >= 8 and right - left >= 8:
            content = content[:, top:bottom, left:right]
            content = TF.resize(content, [final_h, final_w], antialias=True)

        content = self._random_white_balance(content)
        return content

    @staticmethod
    def _random_white_balance(
        content: torch.Tensor, temp_strength: float = 0.12, tint_strength: float = 0.08
    ) -> torch.Tensor:
        """Random color-temperature (warm/cool: R vs. B) and tint (green vs.
        magenta: G vs. R+B) shift, mimicking a camera white-balance error —
        cheaper and more physically-motivated than generic per-channel jitter."""
        temp = random.uniform(-temp_strength, temp_strength)  # + warmer, - cooler
        tint = random.uniform(-tint_strength, tint_strength)  # + magenta, - green

        out = content.float()
        out[0] = out[0] * (1.0 + temp)
        out[2] = out[2] * (1.0 - temp)
        out[1] = out[1] * (1.0 - tint)
        return out.clamp(0, 255).round().to(torch.uint8)
