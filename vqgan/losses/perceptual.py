"""LPIPS perceptual loss between input and reconstruction."""
import torch
import torch.nn as nn

try:
    import lpips
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The `lpips` package is required for perceptual loss. Install it via `uv add lpips`."
    ) from e


class PerceptualLoss(nn.Module):
    """Wraps the `lpips` package's VGG-backbone network. Expects inputs in [-1, 1]."""

    def __init__(self, net: str = "vgg"):
        super().__init__()
        self.lpips_net = lpips.LPIPS(net=net)
        self.lpips_net.eval()
        for param in self.lpips_net.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        return self.lpips_net(x, x_recon).mean()
