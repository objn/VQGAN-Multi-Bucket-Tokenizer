"""Shared building blocks for the encoder/decoder (Taming Transformers style)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def norm(num_channels: int) -> nn.GroupNorm:
    num_groups = 32 if num_channels % 32 == 0 else 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)


def build_2d_sincos_position_embedding(
    height: int, width: int, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Generates a [C, H, W] sinusoidal position embedding on the fly for the given
    grid shape -- no fixed-size table, so it generalizes to any bucket resolution."""
    assert channels % 4 == 0, "channels must be divisible by 4 for 2D sincos position embedding"
    dim_quarter = channels // 4
    omega = torch.arange(dim_quarter, device=device, dtype=torch.float32) / dim_quarter
    omega = 1.0 / (10000 ** omega)

    grid_h = torch.arange(height, device=device, dtype=torch.float32)
    grid_w = torch.arange(width, device=device, dtype=torch.float32)

    out_h = grid_h[:, None] * omega[None, :]  # [H, dim_quarter]
    out_w = grid_w[:, None] * omega[None, :]  # [W, dim_quarter]

    pos_h = torch.cat([out_h.sin(), out_h.cos()], dim=1)  # [H, C/2]
    pos_w = torch.cat([out_w.sin(), out_w.cos()], dim=1)  # [W, C/2]

    pos_h = pos_h[:, None, :].expand(height, width, -1)  # [H, W, C/2]
    pos_w = pos_w[None, :, :].expand(height, width, -1)  # [H, W, C/2]
    pos = torch.cat([pos_h, pos_w], dim=-1)  # [H, W, C]
    return pos.permute(2, 0, 1).to(dtype)  # [C, H, W]


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, dropout: float = 0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = swish(self.norm1(x))
        h = self.conv1(h)
        h = swish(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h


class AttnBlock(nn.Module):
    """Single-head self-attention over spatial positions. Unlike the original VQGAN's
    attention block, this adds a 2D sinusoidal position embedding computed on the fly
    from the actual input H/W, so the same weights generalize across bucket shapes
    instead of relying on a fixed-size positional embedding table."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = norm(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        _, c, height, width = h.shape
        pos = build_2d_sincos_position_embedding(height, width, c, h.device, h.dtype)
        h = h + pos.unsqueeze(0)
        q, k, v = self.q(h), self.k(h), self.v(h)

        b, c, height, width = q.shape
        q = q.reshape(b, c, height * width).permute(0, 2, 1)  # [B, HW, C]
        k = k.reshape(b, c, height * width)  # [B, C, HW]
        attn = torch.bmm(q, k) * (c ** -0.5)  # [B, HW, HW]
        attn = F.softmax(attn, dim=2)

        v = v.reshape(b, c, height * width)  # [B, C, HW]
        attn = attn.permute(0, 2, 1)  # [B, HW(k), HW(q)]
        out = torch.bmm(v, attn)  # [B, C, HW]
        out = out.reshape(b, c, height, width)
        out = self.proj_out(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
