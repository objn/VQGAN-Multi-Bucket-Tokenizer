import torch
import torch.nn as nn

from .refinement import RefinementHead
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
        refine_enabled=False,
        refine_hidden_channels=64,
        refine_num_blocks=2,
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

        # Optional, and off by default so an existing checkpoint keeps exactly
        # the architecture it was trained with. Sees mu and log_b together
        # (2 * out_channels), after they have been laid back out as an image —
        # which is the only place the patch seams exist to be smoothed.
        self.refine = (
            RefinementHead(
                2 * out_channels,
                hidden=refine_hidden_channels,
                num_blocks=refine_num_blocks,
            )
            if refine_enabled
            else None
        )

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

        # x is exactly cat([mu, log_b], dim=1) at this point, so the head gets
        # both halves in one pass; it stays an identity until trained.
        if self.refine is not None:
            x = self.refine(x)

        mu, log_b = x.chunk(2, dim=1)
        recon = 2 * torch.sigmoid(mu) - 1             # [-1, 1], same range as the input
        return recon, mu, log_b

    def last_layer(self) -> nn.Parameter:
        """The final layer's weight, for the adaptive discriminator-weight
        trick (see train_step.py) — the VQGAN/ViT-VQGAN papers balance the
        reconstruction and adversarial losses by comparing how hard *this one
        layer* is pushed by each, rather than fixing their ratio by hand.

        The measuring point stays at `to_pixels` in every mode that trains it,
        so the adaptive weight remains comparable with every run so far. It
        moves only when `to_pixels` is frozen (refine-only training, see
        VQGANTrainConfig.refine.train_stage): the ratio is a quotient of two
        gradients taken at this tensor, and a frozen one has none to take.
        The refinement head's output convolution is then the last trained
        layer, and so the equivalent place to measure."""
        if self.refine is not None and not self.to_pixels.weight.requires_grad:
            return self.refine.last_layer()
        return self.to_pixels.weight
