"""CNN encoder: [B, 3, H, W] -> [B, latent_channels, H/32, W/32] (5 downsampling stages
of /2 each). Fully convolutional -- no assumption that H == W or that either is fixed,
so the same weights handle every bucket resolution in the aspect-ratio bucket table.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from vqgan.models.common import AttnBlock, Downsample, ResnetBlock, norm, swish


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple = (1, 1, 2, 2, 4),
        latent_channels: int = 256,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.grad_checkpointing = grad_checkpointing

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        cur_channels = base_channels
        for mult in channel_mults:
            out_channels = base_channels * mult
            stage = nn.Module()
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(cur_channels, out_channels, dropout=dropout))
                cur_channels = out_channels
            stage.blocks = blocks
            stage.downsample = Downsample(cur_channels)  # every stage downsamples: 5 stages = 32x total
            self.down_blocks.append(stage)

        # mid: resnet -> attn -> resnet, at lowest resolution
        self.mid_block1 = ResnetBlock(cur_channels, cur_channels, dropout=dropout)
        self.mid_attn = AttnBlock(cur_channels)
        self.mid_block2 = ResnetBlock(cur_channels, cur_channels, dropout=dropout)

        self.norm_out = norm(cur_channels)
        self.conv_out = nn.Conv2d(cur_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)

        for stage in self.down_blocks:
            for block in stage.blocks:
                if self.grad_checkpointing and self.training:
                    h = checkpoint.checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)
            h = stage.downsample(h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = swish(self.norm_out(h))
        h = self.conv_out(h)
        return h
