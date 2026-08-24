import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """The "VQ" in VQGAN.

    Supports two codebook update modes, chosen with `use_ema`:
      - use_ema=False : original gradient-based update (codebook_loss backprop)
      - use_ema=True  : EMA / online-k-means style update (no codebook_loss term),
                         plus dead-code revival to prevent codebook collapse
    """

    def __init__(
        self,
        num_embeddings=2048,
        embedding_dim=1024,
        beta=0.25,
        use_ema=True,           # <-- the on/off switch
        ema_decay=0.99,
        ema_eps=1e-5,
        dead_code_threshold=1.0,   # cluster_size below this counts as "unused"
        revive_check_every=50,     # check for dead codes every N forward calls (training only)
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.dead_code_threshold = dead_code_threshold
        self.revive_check_every = revive_check_every

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # Usage tracking (independent of use_ema, which only maintains
        # ema_cluster_size when EMA mode is on) so codebook health can be
        # monitored regardless of the update mode in use.
        self.register_buffer("usage_count", torch.zeros(num_embeddings))

        # EMA buffers are always registered (not just when use_ema=True at
        # construction time) so set_use_ema(True) can turn EMA on later —
        # e.g. after a gradient-based warmup period — without an
        # AttributeError. See set_use_ema() for how they're (re)synced.
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed_avg", self.codebook.weight.data.clone())
        self.register_buffer("_step_count", torch.zeros(1, dtype=torch.long))

        if self.use_ema:
            # EMA codebook entries are NOT updated by the optimizer via gradients,
            # so we freeze them here and update the buffers manually in forward().
            self.codebook.weight.requires_grad_(False)

    def set_use_ema(self, flag: bool):
        """Flip EMA mode on/off at runtime (e.g. warm up with gradient updates,
        then switch to EMA once the encoder has stabilized)."""
        turning_on = flag and not self.use_ema
        self.use_ema = flag
        if flag and self.codebook.weight.requires_grad:
            self.codebook.weight.requires_grad_(False)
        elif not flag and not self.codebook.weight.requires_grad:
            self.codebook.weight.requires_grad_(True)

        if turning_on:
            # Resync EMA state from the current (gradient-trained) codebook
            # instead of whatever it was at __init__ time, so EMA continues
            # smoothly from here rather than snapping back to the initial
            # random codebook.
            self.ema_embed_avg.copy_(self.codebook.weight.data)
            self.ema_cluster_size.zero_()
            self._step_count.zero_()

    def forward(self, z):
        # z: [B, C, H, W] -> [B, H, W, C] -> flatten to [B*H*W, C]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)

        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        token_indices_flat = distances.argmin(dim=1)  # [B*H*W]
        z_q = self.codebook(token_indices_flat).view(z.shape)

        with torch.no_grad():
            self.usage_count.scatter_add_(
                0, token_indices_flat, torch.ones_like(token_indices_flat, dtype=self.usage_count.dtype)
            )

        if self.use_ema and self.training:
            # No-grad + scatter/index_add instead of a one_hot([B*H*W, K]) matmul:
            # this bookkeeping only ever updates buffers (never backprop'd through),
            # but building it via one_hot still (a) allocates a huge [B*H*W, K]
            # intermediate and (b) drags z_flat's full autograd graph along for
            # the ride since z_flat requires grad — wasted compute *and* memory
            # that got big enough (num_embeddings=2048, larger batches) to spill
            # into shared GPU memory and tank throughput once EMA mode kicked in.
            with torch.no_grad():
                z_detached = z_flat.detach()

                batch_cluster_size = torch.zeros(
                    self.num_embeddings, device=z_flat.device, dtype=torch.float32
                )
                batch_cluster_size.scatter_add_(
                    0, token_indices_flat, torch.ones_like(token_indices_flat, dtype=torch.float32)
                )

                batch_embed_sum = torch.zeros(
                    self.num_embeddings, self.embedding_dim, device=z_flat.device, dtype=torch.float32
                )
                batch_embed_sum.index_add_(0, token_indices_flat, z_detached.float())

                self.ema_cluster_size.mul_(self.ema_decay).add_(batch_cluster_size, alpha=1 - self.ema_decay)
                self.ema_embed_avg.mul_(self.ema_decay).add_(batch_embed_sum, alpha=1 - self.ema_decay)

                # Laplace smoothing so cluster sizes never hit exactly zero (avoids div-by-zero)
                n = self.ema_cluster_size.sum()
                smoothed_size = (
                    (self.ema_cluster_size + self.ema_eps)
                    / (n + self.num_embeddings * self.ema_eps) * n
                )
                new_codebook = self.ema_embed_avg / smoothed_size.unsqueeze(1)
                self.codebook.weight.data.copy_(new_codebook)

                self._step_count += 1
                if self.revive_check_every > 0 and self._step_count.item() % self.revive_check_every == 0:
                    self._revive_dead_codes(z_detached)

            # Only a commitment loss is needed in EMA mode (codebook is updated above, not via loss)
            vq_loss = self.beta * F.mse_loss(z_q.detach(), z)
        else:
            # Original gradient-based codebook update
            codebook_loss = F.mse_loss(z_q, z.detach())
            commitment_loss = F.mse_loss(z_q.detach(), z)
            vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator: let gradients flow to the encoder either way
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        token_indices = token_indices_flat.view(z.shape[0], z.shape[1], z.shape[2])
        return z_q, vq_loss, token_indices

    @torch.no_grad()
    def _revive_dead_codes(self, z_flat):
        """Reinitialize codebook entries that are barely ever used with real
        latent vectors sampled from the current batch, so they land somewhere
        the encoder actually produces instead of a meaningless random spot."""
        dead_mask = self.ema_cluster_size < self.dead_code_threshold
        num_dead = int(dead_mask.sum().item())
        if num_dead == 0:
            return

        replacement_idx = torch.randint(0, z_flat.shape[0], (num_dead,), device=z_flat.device)
        # Under AMP, z_flat is bf16/fp16 (encoder runs inside autocast) while the
        # codebook/EMA buffers are plain float32 — indexed assignment (unlike
        # add_/mul_) requires an exact dtype match, so cast explicitly.
        replacements = z_flat[replacement_idx].to(self.codebook.weight.dtype)

        self.codebook.weight.data[dead_mask] = replacements
        self.ema_embed_avg[dead_mask] = replacements
        self.ema_cluster_size[dead_mask] = 1.0  # give it a fresh, small but non-zero count

    def codebook_usage_pct(self) -> float:
        """Fraction of codebook entries used at least once since the last
        reset_usage_stats() call, as a percentage."""
        return 100.0 * (self.usage_count > 0).float().mean().item()

    def reset_usage_stats(self):
        self.usage_count.zero_()
