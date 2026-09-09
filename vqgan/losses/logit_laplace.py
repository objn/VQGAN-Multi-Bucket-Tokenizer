"""Logit-Laplace reconstruction loss (ViT-VQGAN / DALL-E).

Plain L2 treats pixels as unbounded reals, but they are not: they live in a
closed interval, and an L2-trained decoder happily predicts values outside it.
The logit-Laplace likelihood instead maps pixels into (0, 1), models them with
a Laplace distribution *in logit space* — so the support is exactly the valid
range — and lets the decoder predict its own per-pixel uncertainty (`log_b`)
alongside the value (`mu`). Sharp edges, where the decoder should be unsure,
get a wide scale instead of being averaged into mush.

ViT-VQGAN uses it as a 0.1-weighted term next to the 1.0-weighted L2 loss.
"""

import math

import torch

# Pixels are squeezed away from the open endpoints before taking a logit:
# logit(0) and logit(1) are infinite, so a pure black or white pixel would
# otherwise produce an infinite target. 0.1 is the value DALL-E used.
EPS = 0.1


def logit_laplace_nll(mu: torch.Tensor, log_b: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of `target` under Laplace(mu, exp(log_b))
    in logit space.

    `target` is in [-1, 1] (the range the rest of the pipeline uses); `mu` and
    `log_b` are the decoder's raw two-headed output.
    """
    x = (target + 1) / 2                    # [-1, 1] -> [0, 1]
    x = x * (1 - 2 * EPS) + EPS             # -> (EPS, 1 - EPS), safely inside (0, 1)

    # Without a clamp, exp(log_b) can underflow to 0 early in training and the
    # division below goes to inf, which then poisons every gradient in the step.
    log_b = log_b.clamp(-7.0, 7.0)

    return (
        log_b
        + math.log(2.0)
        + torch.log(x)
        + torch.log1p(-x)
        + (torch.logit(x) - mu).abs() / log_b.exp()
    ).mean()
