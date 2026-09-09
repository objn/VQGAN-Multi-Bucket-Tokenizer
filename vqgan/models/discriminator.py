import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator: real vs. fake, per image patch.

    Normalized with GroupNorm, not BatchNorm. BatchNorm's statistics are taken
    over the whole batch, which makes what the discriminator sees — and so how
    hard it pushes back on the generator — a function of batch size: the
    normalization noise that acts as regularization at batch 8 all but vanishes
    at batch 128, and at batch 1 or 2 the statistics are degenerate. It is the
    one place in this pipeline where batch size cannot be compensated for by
    rescaling a constant, so it is normalized within each sample instead
    (32 groups: 4 channels per group at 128, 8 at 256).
    """

    def __init__(self, in_channels=3, base_channels=64, num_groups=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, kernel_size=4, stride=1, padding=1),  # patch-level logits
        )

    def forward(self, x):
        return self.net(x)  # [B, 1, H', W'] - one real/fake score per patch
