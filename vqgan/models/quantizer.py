"""Vector quantization layer with straight-through estimator and optional EMA updates."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Maps continuous encoder features to the nearest codebook entry.

    Codebook size and embedding dim are constructor args so this can be scaled
    (e.g. 16384 -> 65536) without rewriting the class.

    Quantization is per-token and shape-agnostic by construction: forward() flattens
    [B, C, H, W] to [B*H*W, C] for the nearest-neighbor lookup and reshapes back, so it
    makes no assumption about H or W -- it handles every bucket resolution unmodified.
    """

    def __init__(
        self,
        codebook_size: int = 16384,
        codebook_dim: int = 256,
        commitment_beta: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_beta = commitment_beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        if use_ema:
            # EMA codebook: embedding.weight is updated via buffers, not gradients.
            self.embedding.weight.requires_grad_(False)
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_embed_avg", self.embedding.weight.data.clone())

    def forward(self, z: torch.Tensor) -> dict:
        """z: [B, C, H, W] continuous encoder output. Returns dict with quantized output and losses."""
        b, c, h, w = z.shape
        assert c == self.codebook_dim, f"encoder channel dim {c} != codebook_dim {self.codebook_dim}"

        z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)  # [B*H*W, C]

        # squared L2 distance to every codebook entry
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)  # [B*H*W]
        z_q_flat = self.embedding(indices)  # [B*H*W, C]

        if self.use_ema and self.training:
            self._update_ema(z_flat, indices)

        z_q = z_q_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        if self.use_ema:
            codebook_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        else:
            # codebook_loss pulls the codebook (z_q) toward the encoder output (sg[z])
            codebook_loss = F.mse_loss(z_q, z.detach())
        # commitment_loss pulls the encoder output (z) toward the codebook (sg[z_q])
        commitment_loss = F.mse_loss(z, z_q.detach())
        loss = codebook_loss + self.commitment_beta * commitment_loss

        # straight-through estimator
        z_q = z + (z_q - z).detach()

        indices = indices.view(b, h, w)
        perplexity, usage = self._codebook_stats(indices)

        return {
            "z_q": z_q,
            "loss": loss,
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "indices": indices,
            "perplexity": perplexity,
            "codebook_usage": usage,
        }

    @torch.no_grad()
    def _update_ema(self, z_flat: torch.Tensor, indices: torch.Tensor) -> None:
        one_hot = F.one_hot(indices, self.codebook_size).type(z_flat.dtype)  # [N, K]
        cluster_size = one_hot.sum(dim=0)  # [K]
        embed_sum = one_hot.t() @ z_flat  # [K, C]

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_embed_avg.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        smoothed_size = (
            (self.ema_cluster_size + self.ema_epsilon)
            / (n + self.codebook_size * self.ema_epsilon)
            * n
        )
        self.embedding.weight.data.copy_(self.ema_embed_avg / smoothed_size.unsqueeze(1))

    @torch.no_grad()
    def _codebook_stats(self, indices: torch.Tensor) -> tuple:
        flat = indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.codebook_size).float()
        probs = counts / counts.sum()
        nonzero = probs[probs > 0]
        perplexity = torch.exp(-(nonzero * nonzero.log()).sum())
        usage = (counts > 0).float().mean()  # fraction of codebook active in this batch
        return perplexity, usage
