"""ViT-VQGAN (Vector-Quantized GAN with a Vision Transformer autoencoder) in PyTorch.

ViT-VQGAN = ViT Encoder -> factorized, L2-normalized Vector Quantization
(codebook) -> ViT Decoder, trained adversarially with a CNN PatchGAN
discriminator.

Images are never resized: the model works on fixed-size crops taken at native
resolution, and a whole image is reconstructed by running it tile by tile and
blending the tiles back together (see vqgan/data/tiling.py).

Once trained, the Encoder+VectorQuantizer turns any image into a grid of
discrete token indices. Those token sequences are what you would feed into a
separate autoregressive Transformer to learn to generate new sequences.
"""

from .models import (
    VQGAN,
    Decoder,
    Encoder,
    PatchDiscriminator,
    TransformerBlock,
    VectorQuantizer,
    VQOutput,
)
from .training import train_step

__all__ = [
    "TransformerBlock",
    "Encoder",
    "Decoder",
    "VectorQuantizer",
    "PatchDiscriminator",
    "VQGAN",
    "VQOutput",
    "train_step",
]
