"""Confirms the encoder/decoder/quantizer/discriminator stack makes no fixed-shape
assumptions anywhere -- the whole point of the bucket system is that one set of
weights handles every bucket resolution unmodified.

Per .claude/mulit-vqgan.md: "Test this explicitly: run a forward+backward pass
through at least two different buckets in the same test to confirm no shape
assumptions leak in anywhere."
"""
import torch

from vqgan.config import ModelConfig
from vqgan.models.discriminator import PatchGANDiscriminator
from vqgan.models.vqgan import VQGAN

# two different aspect ratios, both dims multiples of 32 (required by the 5x
# downsampling stages) but deliberately not from the production bucket table --
# small enough to run this test on CPU in well under a second
SHAPES = [(1, 3, 128, 128), (2, 3, 96, 160)]


def make_tiny_model() -> VQGAN:
    # tiny channel/codebook sizes -- this test checks shape-agnosticism, not quality,
    # so keep it cheap
    config = ModelConfig(
        base_channels=8,
        channel_mults=(1, 1, 2, 2, 4),
        latent_channels=16,
        codebook_dim=16,
        codebook_size=64,
        num_res_blocks=1,
        discriminator_channels=8,
        discriminator_num_layers=2,
    )
    return VQGAN(config)


def test_forward_backward_across_buckets():
    model = make_tiny_model()
    discriminator = PatchGANDiscriminator(
        in_channels=3, base_channels=8, num_layers=2
    )

    for b, c, h, w in SHAPES:
        x = torch.randn(b, c, h, w)

        out = model(x)
        assert out["reconstruction"].shape == x.shape, (
            f"reconstruction shape {out['reconstruction'].shape} != input shape {x.shape}"
        )
        assert out["indices"].shape == (b, h // 32, w // 32)

        fake_logits = discriminator(out["reconstruction"])
        real_logits = discriminator(x)
        assert fake_logits.shape == real_logits.shape

        loss = out["reconstruction"].mean() + out["codebook_loss"] + fake_logits.mean()
        loss.backward()

        # every parameter that participated should have a gradient -- if any
        # submodule silently ignored a shape (e.g. via a hardcoded reshape), this
        # would show up as unused/None grads instead
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} got no gradient for shape {(h, w)}"

        model.zero_grad(set_to_none=True)
        discriminator.zero_grad(set_to_none=True)


def test_encode_decode_roundtrip_shape():
    model = make_tiny_model()
    model.eval()
    for b, c, h, w in SHAPES:
        x = torch.randn(b, c, h, w)
        indices = model.encode(x)
        assert indices.shape == (b, h // 32, w // 32)
        recon = model.decode_indices(indices)
        assert recon.shape == x.shape
