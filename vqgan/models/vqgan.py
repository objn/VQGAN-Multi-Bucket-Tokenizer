"""Wraps encoder + quantizer + decoder into a single tokenizer module.

Fully convolutional end to end, so one set of weights handles every bucket resolution
in the aspect-ratio bucket table (see vqgan/data/buckets.py) -- there is no
per-resolution construction argument anywhere in this module.
"""
import torch
import torch.nn as nn

from vqgan.config import ModelConfig
from vqgan.models.decoder import Decoder
from vqgan.models.encoder import Encoder
from vqgan.models.quantizer import VectorQuantizer


class VQGAN(nn.Module):
    def __init__(self, config: ModelConfig, grad_checkpointing: bool = False):
        super().__init__()
        self.config = config

        self.encoder = Encoder(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            channel_mults=config.channel_mults,
            latent_channels=config.latent_channels,
            num_res_blocks=config.num_res_blocks,
            grad_checkpointing=grad_checkpointing,
        )
        self.quantizer = VectorQuantizer(
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            commitment_beta=config.commitment_beta,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay,
            dead_code_revival=config.dead_code_revival,
        )
        self.decoder = Decoder(
            out_channels=config.in_channels,
            base_channels=config.base_channels,
            channel_mults=config.channel_mults,
            latent_channels=config.latent_channels,
            num_res_blocks=config.num_res_blocks,
            grad_checkpointing=grad_checkpointing,
        )

    def forward(self, x: torch.Tensor, revive_dead: bool = False) -> dict:
        """x: [B, 3, H, W] in [-1, 1]. Returns dict with reconstruction, quantizer loss/stats, indices.

        revive_dead: passed through to the quantizer -- see VectorQuantizer.forward().
        """
        z = self.encoder(x)
        q_out = self.quantizer(z, revive_dead=revive_dead)
        x_recon = self.decoder(q_out["z_q"])

        return {
            "reconstruction": x_recon,
            "codebook_loss": q_out["loss"],
            "codebook_loss_raw": q_out["codebook_loss"],
            "commitment_loss": q_out["commitment_loss"],
            "indices": q_out["indices"],
            "perplexity": q_out["perplexity"],
            "codebook_usage": q_out["codebook_usage"],
            "num_revived": q_out["num_revived"],
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns discrete token indices [B, H/32, W/32] for downstream AR consumption."""
        z = self.encoder(x)
        return self.quantizer(z)["indices"]

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstructs an image from token indices [B, H, W]."""
        z_q = self.quantizer.embedding(indices)  # [B, H, W, C]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)
