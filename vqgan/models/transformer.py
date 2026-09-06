"""Transformer building blocks shared by the ViT encoder and ViT decoder."""

import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Standard (non-causal) MHSA over the flat patch-token sequence.

    Uses F.scaled_dot_product_attention rather than an explicit softmax(QK^T)V
    so PyTorch can pick a fused/memory-efficient kernel: the token grid here is
    (image_size/patch_size)^2 — 1024 tokens at the defaults — and materializing
    an [B, heads, N, N] score matrix per block is exactly what makes a naive ViT
    autoencoder run out of memory at these sequence lengths.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"model_dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, heads, N, head_dim]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block (norm -> sublayer -> residual), the
    variant ViT-VQGAN uses — pre-LN keeps the residual path clean and trains
    stably without the warmup schedule post-LN needs."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
