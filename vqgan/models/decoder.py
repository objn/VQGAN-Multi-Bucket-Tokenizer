import torch.nn as nn

from .blocks import ResBlock


class Decoder(nn.Module):
    """Quantized latent -> reconstructed image."""

    def __init__(self, out_channels=3, base_channels=64, latent_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, base_channels * 4, kernel_size=3, padding=1),
            ResBlock(base_channels * 4),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1),  # x2
            ResBlock(base_channels * 4),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),  # x4
            ResBlock(base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),  # x8
            ResBlock(base_channels),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z_q):
        return self.net(z_q)
