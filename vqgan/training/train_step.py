import torch
import torch.nn.functional as F

from ..losses import LPIPS_AVAILABLE, get_lpips_model, logit_laplace_nll


def train_step(
    vqgan,
    discriminator,
    opt_g,
    opt_d,
    real_images,
    *,
    l2_weight,
    logit_laplace_weight,
    lpips_weight,
    disc_weight,
    use_lpips,
    amp,
    grad_clip_norm,
    global_step=0,
    disc_start_step=0,
):
    """One generator step + one discriminator step.

    The generator loss is ViT-VQGAN's: L2 + logit-Laplace + LPIPS + hinge GAN,
    weighted 1.0 / 0.1 / 0.1 / 0.1 by default, plus the quantizer's own VQ
    term. Every crop is full of real pixels (tiles are shifted inward at the
    image edge rather than padded), so nothing needs masking out.

    `global_step`/`disc_start_step` implement discriminator warmup: before
    `disc_start_step`, the adversarial term is excluded from the generator
    loss (but the discriminator itself keeps training every step so it isn't
    cold once warmup ends).

    LPIPS is optional and toggled with `use_lpips`. If the `lpips` package
    isn't installed, it's silently skipped even if you asked for it.

    Set `amp=True` to train under bf16 autocast (bf16 has fp32's exponent
    range, so unlike fp16 it doesn't need a GradScaler to avoid overflow).

    `grad_clip_norm` clips each network's gradient norm right before its
    optimizer step, independently for the generator and the discriminator —
    the unbounded hinge/WGAN-style adversarial term can otherwise blow up a
    step's gradients and send the run to NaN.
    """
    device = real_images.device
    device_type = device.type
    effective_disc_weight = disc_weight if global_step >= disc_start_step else 0.0
    lpips_on = use_lpips and LPIPS_AVAILABLE

    # ---- Generator (encoder+quantizer+decoder) step ----
    opt_g.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        out = vqgan(real_images)

        recon_loss = F.mse_loss(out.recon, real_images)
        # The logit-Laplace term is computed in fp32: it takes a log and an
        # exp of the decoder's raw scale head, and bf16 has too few mantissa
        # bits for log1p(-x) near the edges of the pixel range.
        laplace_loss = logit_laplace_nll(out.mu.float(), out.log_b.float(), real_images.float())

        if lpips_on:
            lpips_model = get_lpips_model(device)
            perceptual_loss = lpips_model(out.recon, real_images).mean()  # inputs in [-1, 1]
        else:
            perceptual_loss = torch.tensor(0.0, device=device)

        fake_logits = discriminator(out.recon)
        gan_loss_g = -fake_logits.mean()  # fool the discriminator

        g_loss = (
            l2_weight * recon_loss
            + logit_laplace_weight * laplace_loss
            + out.vq_loss
            + lpips_weight * perceptual_loss
            + effective_disc_weight * gan_loss_g
        )

    g_loss.backward()
    g_grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in vqgan.parameters() if p.requires_grad], grad_clip_norm
    )
    opt_g.step()

    # ---- Discriminator step ----
    opt_d.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        real_logits = discriminator(real_images)
        fake_logits = discriminator(out.recon.detach())
        d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()  # hinge loss

    d_loss.backward()
    d_grad_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_norm)
    opt_d.step()

    return {
        "recon_loss": recon_loss.item(),
        "laplace_loss": laplace_loss.item(),
        "lpips_loss": perceptual_loss.item() if lpips_on else None,
        "vq_loss": out.vq_loss.item(),
        "g_loss": g_loss.item(),
        "d_loss": d_loss.item(),
        "g_grad_norm": g_grad_norm.item(),
        "d_grad_norm": d_grad_norm.item(),
    }
