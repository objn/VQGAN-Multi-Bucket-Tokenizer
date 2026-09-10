import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _distributed() -> bool:
    """True only inside a torchrun-launched process group.

    The codebook is maintained by hand in forward() (and is
    requires_grad=False in EMA mode), so DDP's gradient all-reduce never sees
    it — every cross-rank agreement below has to be arranged explicitly. Read
    from torch.distributed rather than taking a rank/world_size argument so
    every single-process caller (scripts/evaluate.py, reconstruct.py, ...)
    keeps working untouched.
    """
    return dist.is_available() and dist.is_initialized()


class VectorQuantizer(nn.Module):
    """The "VQ" in ViT-VQGAN.

    Two ViT-VQGAN-specific details on top of a plain VQ layer:

      - *Factorized* codes: `embedding_dim` is deliberately small (32) compared
        to the transformer's width (768). The encoder projects down to this
        space before lookup, which is what keeps nearest-neighbour search
        well-conditioned and codebook usage high.
      - *L2-normalized* codes (`l2_normalize=True`): both the encoded latents
        and the codebook entries are projected onto the unit sphere, so squared
        euclidean distance becomes 2 - 2*cosine — lookup is cosine similarity,
        and code magnitude can't drift.

    Supports two codebook update modes, chosen with `use_ema`:
      - use_ema=False : original gradient-based update (codebook_loss backprop)
      - use_ema=True  : EMA / online-k-means style update (no codebook_loss term),
                         plus dead-code revival to prevent codebook collapse
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        beta=0.25,
        l2_normalize=True,      # project latents + codes onto the unit sphere
        use_ema=True,           # <-- the on/off switch
        ema_decay=0.99,
        ema_eps=1e-5,
        dead_code_fraction=0.1,    # "used less than 10% as often as a uniformly-used code"
        revive_check_every=800,    # check for dead codes every N *images* seen (training only)
        reference_batch_size=16,   # the batch size ema_decay/revive_check_every are calibrated at
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.l2_normalize = l2_normalize
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        # A *fraction of uniform usage* rather than an absolute cluster size.
        # ema_cluster_size settles around (tokens per step) / num_embeddings for
        # a healthy codebook, which is 2.0 here (1024 tokens x batch 16 / 8192
        # codes) but was ~7 for the old 128x128 CNN latent grid — so the old
        # absolute threshold of 1.0 goes from "an eighth of average" to "half of
        # average" purely because the architecture changed, and would condemn
        # half of a perfectly healthy codebook every check. Scaling by the mean
        # keeps the criterion the same regardless of token grid, batch size or
        # codebook size, exactly like the per-image EMA decay above.
        self.dead_code_fraction = dead_code_fraction
        self.revive_check_every = revive_check_every

        # ema_decay/revive_check_every used to be applied per *forward call*,
        # which silently made codebook dynamics depend on batch size: at a
        # fixed epoch count, a smaller batch means more steps/epoch, so the
        # EMA "forgets" old batches faster and dead-code revival fires far
        # more often, in epoch-relative terms, than a larger batch — this is
        # exactly what produced very different codebook-usage trajectories
        # on two GPUs training the same dataset at batch_size=1 vs 7. Fixing
        # this by working per-*image* instead: ema_decay is reinterpreted as
        # calibrated at `reference_batch_size` images/step, and the actual
        # per-step decay compounds a per-image rate by however many images
        # were actually in this step (see forward()) — so cumulative decay
        # after N images seen is identical regardless of how those N images
        # were grouped into batches. revive_check_every's default is
        # rescaled the same way (50 steps * 16 images/step, its old
        # implicit assumption) so behavior at the reference batch size is
        # unchanged from before.
        self.ema_decay_per_image = ema_decay ** (1.0 / reference_batch_size)

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        if self.l2_normalize:
            # Start on the unit sphere too, otherwise the first lookups compare
            # unit-norm latents against near-zero codes and every position
            # collapses onto whichever entry happens to be largest.
            self.codebook.weight.data.copy_(F.normalize(self.codebook.weight.data, dim=-1))

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
        # Despite the name (kept for checkpoint compatibility), this counts
        # cumulative *images* seen under EMA mode, not forward calls — see
        # the revival trigger in forward().
        self.register_buffer("_step_count", torch.zeros(1, dtype=torch.long))

        if self.use_ema:
            # EMA codebook entries are NOT updated by the optimizer via gradients,
            # so we freeze them here and update the buffers manually in forward().
            self.codebook.weight.requires_grad_(False)

    def codebook_vectors(self):
        """The codebook as actually used for lookup. Normalizing on read (rather
        than only when writing) means gradient-mode updates stay on the sphere
        too — the raw parameter's magnitude simply stops mattering."""
        if self.l2_normalize:
            return F.normalize(self.codebook.weight, dim=-1)
        return self.codebook.weight

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
        num_images = z.shape[0]
        embed = self.codebook_vectors()

        # z: [B, C, H, W] -> [B, H, W, C] -> flatten to [B*H*W, C]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)
        if self.l2_normalize:
            # Normalize before the lookup and keep the normalized tensor as
            # *the* latent from here on, so the commitment loss and the
            # straight-through estimator both operate in the same space the
            # nearest-neighbour search did.
            z_flat = F.normalize(z_flat, dim=-1)
            z = z_flat.view(z.shape)

        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ embed.t()
            + embed.pow(2).sum(1)
        )
        token_indices_flat = distances.argmin(dim=1)  # [B*H*W]
        z_q = F.embedding(token_indices_flat, embed).view(z.shape)

        # Skipped while being jit-traced (torch.utils.tensorboard.add_graph,
        # torch.onnx.export, ...): an in-place write to a buffer that isn't
        # part of the traced inputs/outputs is a known way to make the
        # tracer lose per-submodule scope info, collapsing what should be a
        # nested Encoder/Quantizer/Decoder graph into a single opaque node.
        # It's a monitoring stat only (codebook_usage_pct()) — irrelevant to
        # a one-off architecture trace anyway.
        if not torch.jit.is_tracing():
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

                # Both are plain sums over this step's tokens, so summing them
                # across ranks is exactly the statistic a single process with
                # the whole global batch would have computed. Without this each
                # rank would run its own codebook off its own 1/N of the data
                # and they would drift apart for the rest of the run. all_reduce
                # hands every rank the identical result, which is also what lets
                # the dead-code check below agree across ranks without a second
                # collective.
                #
                # num_images has to grow with them: it is what converts the
                # per-image decay into this step's decay (and what drives the
                # revival counter), and this step just consumed world_size
                # batches' worth of images, not one. Scaling it here is what
                # keeps "cumulative decay after N images is the same however
                # those N were grouped" true across GPU counts too, not just
                # across batch sizes.
                if _distributed():
                    dist.all_reduce(batch_cluster_size, op=dist.ReduceOp.SUM)
                    dist.all_reduce(batch_embed_sum, op=dist.ReduceOp.SUM)
                    num_images *= dist.get_world_size()

                # Per-image decay compounded by this step's actual image
                # count, not a flat per-step constant — see __init__ for why.
                step_decay = self.ema_decay_per_image ** num_images
                self.ema_cluster_size.mul_(step_decay).add_(batch_cluster_size, alpha=1 - step_decay)
                self.ema_embed_avg.mul_(step_decay).add_(batch_embed_sum, alpha=1 - step_decay)

                # Laplace smoothing so cluster sizes never hit exactly zero (avoids div-by-zero)
                n = self.ema_cluster_size.sum()
                smoothed_size = (
                    (self.ema_cluster_size + self.ema_eps)
                    / (n + self.num_embeddings * self.ema_eps) * n
                )
                new_codebook = self.ema_embed_avg / smoothed_size.unsqueeze(1)
                if self.l2_normalize:
                    new_codebook = F.normalize(new_codebook, dim=-1)
                self.codebook.weight.data.copy_(new_codebook)

                # revive_check_every is in images, but batches don't land
                # exactly on multiples of it — trigger whenever this step's
                # images crossed one (floor-div comparison), not on a
                # modulo, so no batch size skips over a check entirely.
                prev_images_seen = self._step_count.item()
                self._step_count += num_images
                if self.revive_check_every > 0 and (
                    self._step_count.item() // self.revive_check_every
                    > prev_images_seen // self.revive_check_every
                ):
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
        the encoder actually produces instead of a meaningless random spot.

        `z_flat` is already unit-norm when l2_normalize is on, so the
        replacements land on the sphere along with the rest of the codebook.

        Under DDP every rank computes the same `dead_mask` (ema_cluster_size is
        all-reduced in forward()), so all of them arrive here together — but
        the replacements are drawn from *this rank's* latents, and picking a
        real vector is not a statistic that can be averaged the way the EMA
        sums are. So rank 0 picks and broadcasts the result: any other split
        would leave each rank holding a different codebook from here on."""
        threshold = self.dead_code_fraction * self.ema_cluster_size.mean()
        dead_mask = self.ema_cluster_size < threshold
        num_dead = int(dead_mask.sum().item())
        if num_dead == 0:
            return

        distributed = _distributed()
        if not distributed or dist.get_rank() == 0:
            replacement_idx = torch.randint(0, z_flat.shape[0], (num_dead,), device=z_flat.device)
            # Under AMP, z_flat is bf16/fp16 (encoder runs inside autocast) while the
            # codebook/EMA buffers are plain float32 — indexed assignment (unlike
            # add_/mul_) requires an exact dtype match, so cast explicitly.
            replacements = z_flat[replacement_idx].to(self.codebook.weight.dtype)

            # Seed the revived entry right on the "just barely alive" line rather
            # than at an absolute count, for the same scale-independence reason as
            # the threshold itself. ema_embed_avg is a *weighted* sum, so it has to
            # carry the same factor for `ema_embed_avg / cluster_size` to hand back
            # the replacement vector unchanged.
            self.codebook.weight.data[dead_mask] = replacements
            self.ema_cluster_size[dead_mask] = threshold
            self.ema_embed_avg[dead_mask] = replacements * threshold

        if distributed:
            # Whole tensors rather than the masked slices: revival is rare
            # (every revive_check_every images) and these are small, so the
            # simple version costs nothing worth optimizing away.
            for tensor in (self.codebook.weight.data, self.ema_cluster_size, self.ema_embed_avg):
                dist.broadcast(tensor, src=0)

    def codebook_usage_pct(self) -> float:
        """Fraction of codebook entries used at least once since the last
        reset_usage_stats() call, as a percentage."""
        return 100.0 * (self.usage_count > 0).float().mean().item()

    def codebook_used_count(self) -> int:
        """How many codes were used at least once since the last
        reset_usage_stats() call — the same fact as codebook_usage_pct(),
        as a count rather than a percentage."""
        return int((self.usage_count > 0).sum().item())

    def codebook_perplexity(self) -> float:
        """Effective number of codes in use: exp(entropy) of the usage
        distribution, between 1 and num_embeddings.

        Worth reading next to codebook_usage_pct() because that one saturates:
        a codebook where two codes take 90% of every lookup and the remaining
        10% is spread over the other 16,382 still reports ~100% usage, while
        perplexity reports ~2. "Used at least once" and "actually carrying the
        representation" are different questions, and only the second one says
        whether the codebook is really 16k codes wide.

        Reads ema_cluster_size in EMA mode, which is already a running usage
        distribution over recent images, and falls back to usage_count (reset
        at every evaluate()) otherwise.
        """
        counts = self.ema_cluster_size if self.use_ema else self.usage_count
        total = counts.sum()
        if total <= 0:
            return 0.0
        p = counts / total
        p = p[p > 0]  # 0 * log(0) is 0 in the limit, but log(0) is -inf
        entropy = -(p * p.log()).sum()
        return torch.exp(entropy).item()

    def reset_usage_stats(self):
        self.usage_count.zero_()
