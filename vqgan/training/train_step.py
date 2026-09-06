import torch
import torch.nn.functional as F

from ..losses import LPIPS_AVAILABLE, get_lpips_model, logit_laplace_nll


def adaptive_disc_weight(nll_loss, gan_loss, last_layer, eps=1e-4):
    """The VQGAN / ViT-VQGAN adaptive discriminator weight (Esser et al. 2021):

        lambda = ||grad_last_layer[nll_loss]|| / (||grad_last_layer[gan_loss]|| + eps)

    Comparing raw loss values tells you nothing about how hard each one is
    pushing the network — a loss can be numerically small while producing a
    huge gradient, or vice versa. Comparing their gradients *at the decoder's
    last layer* does: it is the ratio that would make the two losses' updates
    to that one layer equal in magnitude, so scaling the adversarial term by
    it keeps reconstruction and adversarial pressure roughly balanced as
    training progresses, instead of a hand-picked ratio that is only right at
    whatever point in training it was tuned at.

    Only the final layer's gradient is needed (not the whole network's), so
    each `torch.autograd.grad` call here backpropagates just from its loss
    down to that one parameter tensor — cheap relative to a full backward.
    `retain_graph=True` on both keeps the graph alive for the real
    `g_loss.backward()` that follows this call.
    """
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]
    weight = nll_grads.norm() / (gan_grads.norm() + eps)
    return weight.clamp(0.0, 1e4).detach()


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

    The generator's reconstruction loss is ViT-VQGAN's L2 + logit-Laplace +
    LPIPS (weighted 1.0 / 0.1 / 0.1 by default) plus the quantizer's own VQ
    term. The adversarial term is *not* added at a fixed weight: following
    Esser et al. (the same trick ViT-VQGAN inherits), it is scaled by
    `adaptive_disc_weight()` so it never dominates or vanishes relative to
    reconstruction regardless of where training is — `disc_weight` is then
    just a final scalar on top of that ratio (0.1 here; the original VQGAN
    paper uses 0.8). Every crop is full of real pixels (tiles are shifted
    inward at the image edge rather than padded), so nothing needs masking
    out.

    `global_step`/`disc_start_step` implement discriminator warmup: before
    `disc_start_step`, the adversarial term is excluded from the generator
    loss entirely (the adaptive-weight gradient calls are skipped too, since
    there is nothing to balance yet), but the discriminator itself keeps
    training every step so it isn't cold once warmup ends.

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
    past_warmup = global_step >= disc_start_step
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

        nll_loss = (
            l2_weight * recon_loss
            + logit_laplace_weight * laplace_loss
            + lpips_weight * perceptual_loss
        )

        fake_logits = discriminator(out.recon)
        gan_loss_g = -fake_logits.mean()  # fool the discriminator

        if past_warmup:
            d_weight = adaptive_disc_weight(nll_loss, gan_loss_g, vqgan.decoder.last_layer())
            effective_disc_weight = d_weight * disc_weight
        else:
            d_weight = torch.tensor(0.0, device=device)
            effective_disc_weight = 0.0

        g_loss = nll_loss + out.vq_loss + effective_disc_weight * gan_loss_g

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
        "d_weight": d_weight.item(),
        "g_loss": g_loss.item(),
        "d_loss": d_loss.item(),
        "g_grad_norm": g_grad_norm.item(),
        "d_grad_norm": d_grad_norm.item(),
    }
