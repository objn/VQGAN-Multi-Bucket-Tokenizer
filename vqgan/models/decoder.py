"""CNN decoder, mirrors the encoder: [B, latent_channels, H/32, W/32] -> [B, 3, H, W].
Fully convolutional -- same weights reconstruct every bucket resolution."""
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from vqgan.models.common import AttnBlock, ResnetBlock, Upsample, norm, swish


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple = (1, 1, 2, 2, 4),
        latent_channels: int = 256,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.grad_checkpointing = grad_checkpointing

        cur_channels = base_channels * channel_mults[-1]

        self.conv_in = nn.Conv2d(latent_channels, cur_channels, kernel_size=3, padding=1)

        self.mid_block1 = ResnetBlock(cur_channels, cur_channels, dropout=dropout)
        self.mid_attn = AttnBlock(cur_channels)
        self.mid_block2 = ResnetBlock(cur_channels, cur_channels, dropout=dropout)

        # mirror the encoder: iterate channel_mults in reverse, upsampling each stage
        self.up_blocks = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            stage = nn.Module()
            blocks = nn.ModuleList()
            # +1 block vs encoder is standard practice, but keep symmetric here: num_res_blocks + 1
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(cur_channels, out_ch, dropout=dropout))
                cur_channels = out_ch
            stage.blocks = blocks
            stage.upsample = Upsample(cur_channels)  # every stage upsamples: 5 stages = 32x total
            self.up_blocks.append(stage)

        self.norm_out = norm(cur_channels)
        self.conv_out = nn.Conv2d(cur_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for stage in self.up_blocks:
            for block in stage.blocks:
                if self.grad_checkpointing and self.training:
                    h = checkpoint.checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)
            h = stage.upsample(h)

        h = swish(self.norm_out(h))
        h = self.conv_out(h)
        return torch.tanh(h)  # output range [-1, 1]
