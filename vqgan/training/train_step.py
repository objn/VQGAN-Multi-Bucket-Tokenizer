import contextlib

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ..losses.lpips_loss import LPIPS_AVAILABLE, get_lpips_model


def _no_sync(module):
    """Skip DDP's gradient all-reduce for a forward/backward whose grads we
    aren't going to step with — e.g. running the discriminator on `recon`
    just to compute the generator's adversarial loss. Those discriminator
    grads get thrown away by opt_d.zero_grad() before the real D step, so
    syncing them across ranks here would just be wasted communication."""
    return module.no_sync() if isinstance(module, DDP) else contextlib.nullcontext()


def train_step(
    vqgan,
    discriminator,
    opt_g,
    opt_d,
    real_images,
    valid_mask=None,
    disc_weight=0.5,
    use_lpips=True,
    lpips_weight=1.0,
    global_step=0,
    disc_start_step=0,
    amp=False,
    grad_clip_norm=1.0,
):
    """One generator step + one discriminator step.

    `valid_mask` ([B,1,H,W], 1=real content / 0=pad margin) confines the L1 and
    LPIPS reconstruction losses to each image's valid region, so the model
    isn't rewarded/penalized for the black padding margin. Pass None to fall
    back to whole-canvas losses.

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

    # ---- Generator (encoder+quantizer+decoder) step ----
    opt_g.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        recon, vq_loss, _ = vqgan(real_images)

        if valid_mask is not None:
            diff = (recon - real_images).abs() * valid_mask
            num_valid = (valid_mask.sum() * real_images.shape[1]).clamp(min=1)
            recon_loss = diff.sum() / num_valid
        else:
            recon_loss = F.l1_loss(recon, real_images)

        if use_lpips and LPIPS_AVAILABLE:
            lpips_model = get_lpips_model(device)
            lpips_map = lpips_model(recon, real_images)  # [B,1,H,W], expects inputs in [-1, 1]
            if valid_mask is not None:
                perceptual_loss = (lpips_map * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            else:
                perceptual_loss = lpips_map.mean()
        else:
            perceptual_loss = torch.tensor(0.0, device=device)

        fake_logits = discriminator(recon)
        gan_loss_g = -fake_logits.mean()  # fool the discriminator

        g_loss = (
            recon_loss + vq_loss + lpips_weight * perceptual_loss + effective_disc_weight * gan_loss_g
        )

    # discriminator.no_sync(): this backward also produces grads for the
    # discriminator's parameters (fake_logits depends on them), but only
    # opt_g.step() runs after it — those discriminator grads get discarded
    # by opt_d.zero_grad() below, so skip syncing them across ranks here.
    with _no_sync(discriminator):
        g_loss.backward()
    g_grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in vqgan.parameters() if p.requires_grad], grad_clip_norm
    )
    opt_g.step()

    # ---- Discriminator step ----
    opt_d.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        # One forward call on cat([real, fake]) rather than two separate
        # calls: two BatchNorm forward passes before a single backward() is
        # a known DDP + BatchNorm trap (each forward bumps running_mean/var
        # in place, and the second bump can invalidate a tensor version the
        # first forward's graph saved for backward — "modified by an
        # inplace operation" under DDP even though single-GPU tolerates it).
        combined_logits = discriminator(torch.cat([real_images, recon.detach()], dim=0))
        real_logits, fake_logits = combined_logits.chunk(2, dim=0)
        d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()  # hinge loss

    d_loss.backward()
    d_grad_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_norm)
    opt_d.step()

    return {
        "recon_loss": recon_loss.item(),
        "lpips_loss": perceptual_loss.item() if use_lpips and LPIPS_AVAILABLE else None,
        "vq_loss": vq_loss.item(),
        "g_loss": g_loss.item(),
        "d_loss": d_loss.item(),
        "g_grad_norm": g_grad_norm.item(),
        "d_grad_norm": d_grad_norm.item(),
    }
