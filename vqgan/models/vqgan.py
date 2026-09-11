from typing import NamedTuple

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .quantizer import VectorQuantizer


class VQOutput(NamedTuple):
    """What a VQGAN forward pass produces.

    A named tuple rather than a bare 5-tuple because most call sites want one
    or two of these (`evaluate` only needs `recon`, the loss needs `mu`/`log_b`)
    and positional unpacking with a row of underscores is easy to get wrong.
    """

    recon: torch.Tensor          # [B, 3, H, W] in [-1, 1]
    mu: torch.Tensor             # [B, 3, H, W] logit-space location
    log_b: torch.Tensor          # [B, 3, H, W] logit-Laplace log-scale
    vq_loss: torch.Tensor        # scalar
    token_indices: torch.Tensor  # [B, H/patch, W/patch] long


class VQGAN(nn.Module):
    """Full ViT-VQGAN generator: ViT encoder + vector quantizer + ViT decoder.

    The adversarial half of "GAN" lives in PatchDiscriminator, which is still a
    CNN — only the autoencoder became a transformer.
    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        model_dim,
        depth,
        num_heads,
        mlp_ratio,
        code_dim,
        num_embeddings,
        use_ema=True,
        refine_enabled=False,
        refine_hidden_channels=64,
        refine_num_blocks=2,
    ):
        super().__init__()
        tower = dict(
            image_size=image_size,
            patch_size=patch_size,
            model_dim=model_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            code_dim=code_dim,
        )
        self.encoder = Encoder(**tower)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=code_dim, use_ema=use_ema
        )
        # Decoder-only: the refinement head runs after unpatchify, so there is
        # no mirror of it on the encoder side. Defaulted off and accepted as
        # plain kwargs so a checkpoint written before the head existed loads
        # with VQGAN(**ckpt["model_config"]) unchanged.
        self.decoder = Decoder(
            **tower,
            refine_enabled=refine_enabled,
            refine_hidden_channels=refine_hidden_channels,
            refine_num_blocks=refine_num_blocks,
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, token_indices = self.quantizer(z)
        recon, mu, log_b = self.decoder(z_q)
        return VQOutput(recon, mu, log_b, vq_loss, token_indices)

    @torch.no_grad()
    def encode(self, x):
        """Image -> discrete token grid [B, H/patch, W/patch]."""
        return self.quantizer(self.encoder(x))[2]

    @torch.no_grad()
    def decode(self, token_indices):
        """Token grid -> image. Inverse of encode()."""
        embed = self.quantizer.codebook_vectors()
        z_q = torch.nn.functional.embedding(token_indices, embed)  # [B, gh, gw, code_dim]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)[0]
