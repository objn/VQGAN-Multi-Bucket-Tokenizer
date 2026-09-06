import torch
import torch.nn as nn

from .transformer import TransformerBlock


class Decoder(nn.Module):
    """ViT-VQGAN decoder: quantized code grid -> reconstructed image.

    Mirror image of the encoder — the low-dimensional codes are projected back
    up to `model_dim`, run through `depth` transformer blocks, and each output
    token is linearly mapped to the pixels of its own patch (the "unpatchify"
    step), which are then stitched back into a full image.

    The output head emits *two* values per pixel rather than one: `mu` and
    `log_b`, the location and log-scale of a logit-Laplace distribution over
    that pixel (see vqgan/losses/logit_laplace.py). `mu` lives in logit space,
    so squashing it with a sigmoid is what turns it back into an image — that
    sigmoid is why there's no separate Tanh layer here.
    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        model_dim,
        depth,
        num_heads,
        mlp_ratio,
        code_dim,
        out_channels=3,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.grid_size = image_size // patch_size

        self.from_code = nn.Linear(code_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size**2, model_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(model_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(model_dim)
        # 2 * out_channels: mu and log_b for every pixel of the patch.
        self.to_pixels = nn.Linear(model_dim, patch_size * patch_size * out_channels * 2)

    def forward(self, z_q):
        b, _, gh, gw = z_q.shape
        x = z_q.flatten(2).transpose(1, 2)            # [B, N, code_dim]
        x = self.from_code(x) + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.to_pixels(self.norm(x))              # [B, N, p*p*2C]

        p, c = self.patch_size, self.out_channels
        x = x.view(b, gh, gw, p, p, 2 * c)
        x = x.permute(0, 5, 1, 3, 2, 4)               # [B, 2C, gh, p, gw, p]
        x = x.reshape(b, 2 * c, gh * p, gw * p)

        mu, log_b = x.chunk(2, dim=1)
        recon = 2 * torch.sigmoid(mu) - 1             # [-1, 1], same range as the input
        return recon, mu, log_b
