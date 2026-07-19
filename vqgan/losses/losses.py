"""Combines reconstruction + codebook + perceptual + adversarial losses per the training schedule."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vqgan.config import LossConfig
from vqgan.losses.adversarial import discriminator_loss, generator_adv_loss
from vqgan.losses.perceptual import PerceptualLoss


def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """L1 loss between input and reconstruction (not L2 -- L2 over-smooths)."""
    return F.l1_loss(x_recon, x)


class VQGANLoss(nn.Module):
    """Computes the full generator loss and the discriminator loss.

    Phase A (step < discriminator_start_step): recon + codebook + perceptual only.
    Phase B (step >= discriminator_start_step): adds the adversarial term.
    """

    def __init__(self, config: LossConfig, perceptual_net: str = "vgg"):
        super().__init__()
        self.config = config
        self.perceptual_loss = PerceptualLoss(net=perceptual_net)

    def discriminator_active(self, step: int) -> bool:
        return step >= self.config.discriminator_start_step

    def generator_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        codebook_loss: torch.Tensor,
        step: int,
        fake_logits: torch.Tensor | None = None,
    ) -> dict:
        l_recon = reconstruction_loss(x, x_recon)
        l_perceptual = self.perceptual_loss(x, x_recon)

        l_adv = torch.zeros((), device=x.device, dtype=l_recon.dtype)
        if self.discriminator_active(step):
            assert fake_logits is not None, "fake_logits required once discriminator is active"
            l_adv = generator_adv_loss(fake_logits, loss_type=self.config.adv_loss_type)

        total = (
            l_recon
            + codebook_loss
            + self.config.lambda_perceptual * l_perceptual
            + self.config.lambda_adv * l_adv
        )

        return {
            "total": total,
            "recon_loss": l_recon.detach(),
            "perceptual_loss": l_perceptual.detach(),
            "adversarial_loss": l_adv.detach(),
        }

    def discriminator_loss_fn(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        return discriminator_loss(real_logits, fake_logits, loss_type=self.config.adv_loss_type)
