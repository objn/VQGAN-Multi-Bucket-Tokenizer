"""Post-decoder convolutional refinement head.

The ViT decoder maps each token to the pixels of *its own* patch and nothing
else: two neighbouring patches are written by two independent linear heads that
never see each other's output, so the seam between them is free to jump. That
shows up as a faint 8x8 grid over reconstructions. It is a different seam from
the one vqgan/data/tiling.py blends — that one is between 256px *tiles* of a
large image, this one is between patches inside a single forward pass — and no
amount of tile stitching can reach it.

A convolution can: a stack of 3x3 layers over the assembled image has a
receptive field spanning several patches, so it sees both sides of every seam
at once.
"""

import math

import torch
import torch.nn as nn


def _num_groups(channels: int) -> int:
    """GroupNorm's group count, at most 8 and always a divisor of `channels`.

    GroupNorm rather than BatchNorm for the same reason PatchDiscriminator
    switched: batch size at eval/inference time can be 1, where batch
    statistics are degenerate, and it would make the head's behaviour depend
    on how many crops happen to be in the batch.
    """
    return math.gcd(8, channels)


class ResBlock(nn.Module):
    """Pre-norm 3x3 conv x2 with a residual connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_num_groups(channels), channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_num_groups(channels), channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class RefinementHead(nn.Module):
    """Smooths the assembled image after unpatchify.

    Runs on the decoder's raw output — both halves of it, `mu` and `log_b`
    concatenated on the channel axis, not just the visible reconstruction.
    `log_b` is written patch-by-patch by the same linear head and carries the
    same seams; leaving it raw would not show up in a preview image but it
    would still feed a discontinuous per-pixel scale into the logit-Laplace
    term, so the two are refined together.

    The final convolution is zero-initialized, which makes the whole module an
    exact identity at step 0: adding it to a checkpoint that has already seen
    millions of images changes its outputs by precisely nothing until the head
    is actually trained.
    """

    def __init__(self, channels: int, hidden: int = 64, num_blocks: int = 2):
        super().__init__()
        self.in_conv = nn.Conv2d(channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(hidden, channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        h = self.in_conv(x)
        for block in self.blocks:
            h = block(h)
        return x + self.out_conv(h)

    def last_layer(self) -> nn.Parameter:
        """The head's output weight — what the adaptive discriminator weight is
        measured at while the decoder proper is frozen. See
        Decoder.last_layer()."""
        return self.out_conv.weight
