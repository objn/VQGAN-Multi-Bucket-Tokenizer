import torch.nn as nn

from .blocks import ResBlock


class Encoder(nn.Module):
    """Image (H, W, 3) -> latent feature map (H/8, W/8, latent_dim)."""

    def __init__(self, in_channels=3, base_channels=64, *, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            ResBlock(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # /2
            ResBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),  # /4
            ResBlock(base_channels * 4),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1),  # /8
            ResBlock(base_channels * 4),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, latent_dim, kernel_size=1),  # project to latent_dim
        )

    def forward(self, x):
        return self.net(x)  # [B, latent_dim, H/8, W/8]
