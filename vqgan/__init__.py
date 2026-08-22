"""Simplified VQGAN (Vector-Quantized GAN) in PyTorch.

VQGAN = CNN Encoder -> Vector Quantization (codebook) -> CNN Decoder,
trained adversarially with a CNN PatchGAN discriminator.

Once trained, the Encoder+VectorQuantizer turns any image into a grid of
discrete token indices. Those token sequences are what you'd feed into a
separate autoregressive Transformer to learn to generate new sequences.
"""

from .models import VQGAN, Decoder, Encoder, PatchDiscriminator, ResBlock, VectorQuantizer
from .training import train_step

__all__ = [
    "ResBlock",
    "Encoder",
    "Decoder",
    "VectorQuantizer",
    "PatchDiscriminator",
    "VQGAN",
    "train_step",
]
