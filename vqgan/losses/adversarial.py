"""Generator and discriminator adversarial loss terms (hinge or non-saturating)."""
import torch
import torch.nn.functional as F


def generator_adv_loss(fake_logits: torch.Tensor, loss_type: str = "hinge") -> torch.Tensor:
    """Loss applied to the generator (encoder+decoder+quantizer) to fool the discriminator."""
    if loss_type == "hinge":
        return -fake_logits.mean()
    if loss_type == "non_saturating":
        return F.softplus(-fake_logits).mean()
    raise ValueError(f"Unknown adv_loss_type: {loss_type}")


def discriminator_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor, loss_type: str = "hinge"
) -> torch.Tensor:
    """Standard discriminator loss: push real logits up, fake logits down."""
    if loss_type == "hinge":
        loss_real = F.relu(1.0 - real_logits).mean()
        loss_fake = F.relu(1.0 + fake_logits).mean()
        return 0.5 * (loss_real + loss_fake)
    if loss_type == "non_saturating":
        loss_real = F.softplus(-real_logits).mean()
        loss_fake = F.softplus(fake_logits).mean()
        return 0.5 * (loss_real + loss_fake)
    raise ValueError(f"Unknown adv_loss_type: {loss_type}")


def adaptive_discriminator_weight(
    recon_loss_grad: torch.Tensor, adv_loss_grad: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    """Optional VQGAN-paper trick: scale lambda_adv by the ratio of gradient norms
    w.r.t. the decoder's last layer, so the adversarial loss doesn't dominate/destabilize.
    Not used by default (config.lambda_adv is a fixed scalar), but available for tuning.
    """
    recon_norm = torch.norm(recon_loss_grad)
    adv_norm = torch.norm(adv_loss_grad)
    weight = recon_norm / (adv_norm + eps)
    return torch.clamp(weight, 0.0, 1e4).detach()
