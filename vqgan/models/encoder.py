import torch
import torch.nn as nn

from .transformer import TransformerBlock


class Encoder(nn.Module):
    """ViT-VQGAN encoder: image (H, W, 3) -> one code vector per patch.

    Image -> non-overlapping patch_size x patch_size patches -> linear patch
    embedding -> learned position embeddings -> `depth` transformer blocks ->
    linear projection down to `code_dim`.

    That last projection is the paper's *factorized* code: attention runs at
    the full `model_dim` (768), but codebook lookup happens in a deliberately
    low-dimensional space (32) where nearest-neighbour search is well-behaved
    and codebook usage stays high, instead of quantizing 768-d vectors
    directly.

    Output is shaped [B, code_dim, H/p, W/p] so the quantizer, the masking in
    train_step, and the decoder can all keep treating the latent as a spatial
    grid rather than a flat sequence.
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
        in_channels=3,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size

        # A stride-p, kernel-p conv *is* the per-patch linear embedding, and it
        # does the patch extraction in the same op.
        self.patch_embed = nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size**2, model_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(model_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.to_code = nn.Linear(model_dim, code_dim)

    def forward(self, x):
        x = self.patch_embed(x)                       # [B, model_dim, gh, gw]
        b, _, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)              # [B, N, model_dim]
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.to_code(self.norm(x))                # [B, N, code_dim]
        return x.transpose(1, 2).reshape(b, -1, gh, gw)  # [B, code_dim, gh, gw]
