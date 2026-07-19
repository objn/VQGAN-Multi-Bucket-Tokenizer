"""PatchGAN discriminator: classifies overlapping patches as real/fake instead of the
whole image. Fully convolutional (no Linear/Flatten), so it scores patches from any
bucket resolution with the same weights."""
import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_layers: int = 3):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = base_channels
        for i in range(1, num_layers):
            prev_channels = channels
            channels = min(base_channels * (2 ** i), 512)
            layers += [
                nn.Conv2d(prev_channels, channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        prev_channels = channels
        channels = min(base_channels * (2 ** num_layers), 512)
        layers += [
            nn.Conv2d(prev_channels, channels, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        layers += [nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a map of per-patch real/fake logits, e.g. [B, 1, H', W']."""
        return self.model(x)
