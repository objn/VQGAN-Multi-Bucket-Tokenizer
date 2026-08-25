import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .quantizer import VectorQuantizer


class VQGAN(nn.Module):
    """Full VQGAN wrapper: encoder + vector quantizer + decoder."""

    def __init__(self, latent_dim, num_embeddings, use_ema=True):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=latent_dim, use_ema=use_ema
        )
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, token_indices = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, token_indices
