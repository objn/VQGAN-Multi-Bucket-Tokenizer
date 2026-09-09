import math
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Inputs to scripts/build_index.py — which files exist and how they split.

    Everything about how those files are *read* during training (crop size,
    shuffle buffer, ...) lives in VQGANTrainConfig instead, so there is exactly
    one place to change each knob.
    """

    manifest_dir: str = "images-parquet"   # *.json pointers to parquet datasets on other disks
    folder_root: str = "images"            # drop your own images here
    index_path: str = "data/index.json"

    # Images shorter than this on either side are left out of the index
    # entirely: the model crops at native resolution and never upscales, so
    # they have nothing to contribute. Keep it equal to
    # VQGANTrainConfig.tile_size. Only enforceable on `folder_root` — see
    # ParquetShard for why the parquet datasets are filtered at training time.
    min_size: int = 256

    # Only applies to `folder_root`; the parquet datasets ship their own splits.
    val_frac: float = 0.05
    test_frac: float = 0.05
    seed: int = 0


@dataclass
class VQGANTrainConfig:
    index_path: str = "data/index.json"
    checkpoint_dir: str = "checkpoints"
    out_dir: str = "outputs/vqgan"
    resume: str = ""  # path to a checkpoint saved by this script; "" = train from scratch

    # ViT-VQGAN architecture. tile_size/patch_size fix the token grid at
    # (tile_size/patch_size)^2 = 1024 tokens, which is what the paper trains at
    # and what fits comfortably in 12GB.
    tile_size: int = 256
    patch_size: int = 8
    model_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    code_dim: int = 32            # factorized: lookup happens in this space, not model_dim
    num_embeddings: int = 16384

    # Measured on a 12GB RTX 3080 Ti, at tile_size 256 with LPIPS on. Peak
    # memory reserved by torch, before the ~1.5GB the desktop already holds:
    #
    #   batch  4 ->  4.8GB   27.2 crops/s
    #   batch  8 ->  7.8GB   30.3 crops/s
    #   batch 12 -> 10.5GB   31.4 crops/s   (12.0GB with the desktop: too close)
    #   batch 16 -> 13.5GB    0.3 crops/s   (over the card; on Windows the WDDM
    #                                        driver spills to host RAM over PCIe
    #                                        instead of raising OutOfMemoryError,
    #                                        so this does not crash, it just runs
    #                                        180x slower)
    #
    # Throughput has already plateaued by 8 — this is compute-bound, not
    # batch-bound, so a larger batch buys ~3% for 2.7GB and a spill risk. On a
    # bigger card, raise it freely: it is a pure throughput knob — every
    # *_steps field below is already a batch-invariant count of images (see
    # the "Schedule" section), so a bigger batch does not change what a run
    # trains on, only how fast it gets there.
    batch_size: int = 8
    # The batch size the learning rate below was tuned at (see scaled_lr()).
    # Nothing else reads it — the *_steps schedule fields below don't need a
    # reference batch, since they're already batch-invariant image counts.
    # Changing it reinterprets the learning rate alone.
    #
    # VectorQuantizer keeps its own anchor of the same kind (a
    # reference_batch_size for ema_decay, see quantizer.py) — a separate knob
    # for the codebook's own calibration, unrelated to this one.
    reference_batch_size: int = 8
    # The loader streams ~1100 crops/s with 6 workers against a GPU that eats
    # ~30, so it is nowhere near the bottleneck; 4 keeps a 6x margin while
    # holding ~3GB less host RAM in parquet buffers.
    num_workers: int = 4
    # Every image contributes its *whole* tile grid, so there is no crops-per-image
    # cap; a big photo just yields more tiles. Given as a fraction of tile_size,
    # so the grid keeps its shape if tile_size changes, and it sets two things at
    # once: how much neighbouring training tiles overlap, and how far the grid is
    # randomly nudged each pass (half the overlap — see jittered_tile_origins).
    # 0 therefore means no overlap *and* no jitter: the same crops every epoch.
    #
    # Measured over the first 4,000 images of ImageNet train shard 0 (3,607 of
    # them big enough to crop; the other 9.8% are under 256px on a side and are
    # dropped):
    #
    #   plan_tiles, overlap 0   ->  5.13 tiles/image (median 4)
    #   jittered, ratio 0.15    ->  2.98 tiles/image (median 2)
    #
    # The drop is not lost coverage. plan_tiles pulled its last tile back to the
    # image edge, which on the typical 500x375 photo added a whole extra row and
    # column overlapping the previous ones by ~50% — the same pixels, cropped
    # twice. The jittered grid drops those duplicates and recovers the variety a
    # different way, by moving every tile a little on each pass.
    tile_overlap_ratio: float = 0.15
    # Reservoir size in crops. What it defends against is the shape of the
    # stream: CropDataset yields one photo's whole tile grid before moving to
    # the next, so consecutive crops are the same scene under the same light,
    # overlapping each other by tile_overlap_ratio. Measured on ImageNet train
    # shard 0, an unbuffered batch of 8 holds 3.5 distinct photos on average
    # and sometimes 1 — a batch that is smaller than it looks.
    #
    # Raised to at least 8x batch_size at run time, because what matters is how
    # many batches the buffer can interleave, not how many crops it holds: 1024
    # crops is 128 batches' worth of mixing at batch 8 and 8 batches' worth at
    # batch 128.
    #
    # Not about class ordering, though it would be a good reason if it were
    # true: the shards are already shuffled (639 distinct labels in the first
    # 1,000 rows of train shard 0, and no two adjacent rows share one).
    shuffle_buffer: int = 1024
    # Which corpus to draw from. "parquet" is the downloaded dataset with its
    # own train/validation/test division (pretraining); "folder" is images/,
    # split by the ratios given to build_index.py (finetuning). They are kept
    # separate so pretraining never silently consumes the finetuning set.
    source: str = "parquet"

    # A time budget for the loss curve drawn *during* training, not a property
    # of the validation set. The validation split is whatever the dataset says
    # it is and is consumed whole by scripts/evaluate.py; at 50,000 ImageNet
    # validation images (~257k tiles) one full pass takes ~2.5 hours at this
    # model size, against ~9 minutes of training between evals, so the
    # in-training readout walks a fixed, size-balanced subset instead (see
    # eval_size_groups below) — computed once per run, not re-sampled every
    # evaluation call, so the curve stays comparable point-to-point.
    #
    # A floor, in crops. The run uses max(eval_images, 64 * batch_size), so a
    # machine big enough for a large batch spends its speed on a less noisy
    # readout: 2,048 crops up to batch 32, 8,192 at batch 128. Note what that
    # costs and what it gives up:
    #
    #   - it costs training time. Measured end to end (2,048 crops, bf16,
    #     including the loader's worker startup): 30.3s, against ~528s of
    #     training per 16,000 images at batch 8 — 5.7% of the run. The share
    #     grows with the batch, because the readout grows and the interval
    #     between readouts does not: ~23% at batch 128. Lower
    #     eval_every_steps' twin, or pin --eval-images, if that is too much.
    #   - the prefix length changes the number itself. Measured on the 40k
    #     checkpoint: 0.14689 over 512 crops, 0.14863 over 1,024, 0.14738 over
    #     2,048, 0.14476 over 4,096 — a spread of 0.004, which is real
    #     training progress' worth. Val curves are therefore comparable
    #     between runs at the same batch size, not across different ones.
    #     Pin --eval-images above 64 * batch_size when comparing runs that
    #     used different batches.
    eval_images: int = 8_192
    # How the eval_images crops above are *chosen*, not how many. A
    # sequential (or plain random) prefix of the validation split is badly
    # skewed by native image size — measured on ImageNet-1k validation,
    # ~84% of croppable images are the ~500x375 "web-resized" convention
    # worth exactly 2 crops each, while images with a shorter side >=768px
    # (up to 3,646px, one alone worth 352 crops) are a ~2% scattered tail a
    # short prefix essentially never reaches.
    #
    # Not fixed pixel breakpoints: this is the *count* of log-spaced size
    # groups, computed fresh each run from the real min/max shorter side
    # found in the validation split (compute_log_size_edges() in
    # vqgan/data/eval_subset.py) — so it stays correct as the dataset
    # changes rather than hardcoding boundaries tuned to one corpus.
    # build_balanced_val_subset() then round-robins one uniformly random
    # image per group, largest group first, until eval_images is filled —
    # spending the budget across the whole size range instead of mostly on
    # 2-crop images. Seeded with `seed` below, computed once at startup, so
    # the exact same crops are reused at every evaluation call in a run.
    eval_size_groups: int = 5
    # "" = build the eval subset above fresh at startup (two header-only
    # passes over the validation split, then decode the selection — see
    # eval_images/eval_size_groups above). On ImageNet that scan takes real
    # wall-clock time before the first training step, and it produces the
    # exact same crops every time given the same source/tile_size/
    # tile_overlap_ratio/eval_images/eval_size_groups/seed — so it only ever
    # needs doing once. Point this at a cache written by
    # scripts/prep_eval_data.py to load that decoded selection straight from
    # disk instead. train_vqgan.py refuses a cache built from different
    # settings rather than silently evaluating on the wrong subset.
    eval_prep_file: str = ""

    # ---- Schedule, in steps at batch_size=1 ----
    #
    # Not in optimizer steps at whatever batch_size this run actually uses,
    # and not named "images" even though that's what these counts are: at
    # batch_size=1 a step consumes exactly one image, so a field written here
    # is simultaneously "how many steps to run at batch 1" and "how much data
    # to train on" — the same number either way. Raising batch_size does not
    # change that number. It changes how many optimizer steps it takes to get
    # through it (value // batch_size — fewer, bigger steps) and how fast
    # each one runs, but warmups, eval, checkpoints and the LR decay all still
    # land at the same point in the data. batch_size is therefore a pure
    # throughput knob: a bigger batch trains on exactly the same schedule,
    # just faster.
    #
    # Epochs are still too coarse a unit to schedule on: one pass over
    # ImageNet-1k is ~3.8M crops at the current tiling.
    max_steps: int = 3_400_000
    # 0 = EMA codebook updates from the very first step. The old pipeline warmed
    # up with gradient-based updates first, on the theory that EMA from step 0
    # locks in a noisy encoder — but measured over 1500 steps that warmup is
    # actively harmful here: gradient updates alone never break the initial
    # index collapse (codebook usage sat at 0.0-0.1% for the whole warmup, so
    # the encoder was training through a one-code bottleneck and val L1 *rose*,
    # 0.377 -> 0.490), and only EMA plus dead-code revival pulled it out. With
    # warmup off, val L1 at step 250 already beat what the warmed-up run reached
    # at step 1500 (0.238 vs 0.233 at 6x the steps) and codebook usage ended
    # higher (61% vs 52%). Set this above 0 to get the old behavior back.
    ema_warmup_steps: int = 10_000
    disc_warmup_steps: int = 20_000   # images before adversarial loss contributes to g_loss
    # 0 = no LR warmup, cosine decay starts at base_lr on image 0 (the old
    # behavior). Above 0, lr ramps linearly from 0 up to base_lr over this
    # many images, then the cosine decay in cosine_lr() takes over from
    # base_lr down to min_lr over the images remaining until max_steps.
    lr_warmup_steps: int = 10_000
    eval_every_steps: int = 10_000
    checkpoint_every_steps: int = 100_000
    log_every_steps: int = 10_000

    # Learning rate at reference_batch_size. The rate actually used is this
    # scaled by lr_scaling, because a larger batch averages away gradient noise
    # and a rate tuned against the noisier gradient then understeps:
    #
    #   "sqrt"   lr * sqrt(batch / reference)   batch 128 -> 4.0e-4
    #   "linear" lr * (batch / reference)       batch 128 -> 1.6e-3
    #   "none"   lr, whatever the batch is
    #
    # sqrt by default. The linear rule is the SGD result (Goyal et al.); Adam
    # already divides by a running gradient magnitude, so only the *noise* term
    # is left to correct for, which grows as sqrt(batch) — and 1.6e-3 is well
    # into the range where a ViT plus an adversarial term goes unstable.
    lr: float = 1e-4
    lr_scaling: str = "sqrt"
    min_lr: float = 1e-6
    end_steps_lr: float = 3_400_000  # images at which lr decays to min_lr

    # ViT-VQGAN loss weights.
    l2_weight: float = 1.0
    logit_laplace_weight: float = 0.1
    lpips_weight: float = 1.0
    # Final scalar on top of the *adaptive* discriminator weight (Esser et
    # al.'s ||grad nll|| / ||grad gan|| ratio at the decoder's last layer —
    # see train_step.adaptive_disc_weight), not a fixed contribution to
    # g_loss by itself.
    #
    # This started at 0.1, the figure in ViT-VQGAN's loss table — but that
    # table is for a *fixed* GAN weight, and the adaptive ratio already
    # normalizes the adversarial gradient against the reconstruction one, so
    # 0.1 discounted it a second time. A 44k-step run measured lambda ~0.03,
    # making the adversarial term's effective weight ~0.003: it had
    # essentially no say, and reconstructions stayed blocky at the 8x8 patch
    # boundaries the adversarial term is what teaches the decoder to smooth
    # over. Esser et al., whose ratio this is, use 0.75-0.8 for the same
    # scalar; 1.0 here is at the top of that range.
    disc_weight: float = 1.25
    use_lpips: bool = True

    amp: bool = True            # autocast + train in bf16 (no GradScaler needed for bf16)
    # Every loss here is mean-reduced, so the *expected* gradient norm does not
    # depend on batch size and 1.0 keeps its meaning. Its variance does shrink
    # with a larger batch, though, so clipping fires less often — one of the
    # two places batch size still leaks into training, along with Adam's betas
    # (a momentum window measured in steps; see the optimizer in
    # scripts/train_vqgan.py). Both are left fixed on purpose.
    grad_clip_norm: float = 1.0

    seed: int = 24

    def scaled_lr(self) -> float:
        """The learning rate to actually train at, given batch_size.

        Kept here rather than in the training loop so the number a run uses is
        derivable from its config alone — a checkpoint's config is the only
        record of what it was trained at.
        """
        ratio = self.batch_size / self.reference_batch_size
        if self.lr_scaling == "none":
            return self.lr
        if self.lr_scaling == "linear":
            return self.lr * ratio
        if self.lr_scaling == "sqrt":
            return self.lr * math.sqrt(ratio)
        raise ValueError(f"lr_scaling must be sqrt, linear or none, got {self.lr_scaling!r}")

    def model_config(self) -> dict:
        """The subset of this config that defines the network's shape.

        Stored inside every checkpoint so a checkpoint can be rebuilt without
        guessing at whatever the config defaults happen to be at load time.
        """
        return {
            "image_size": self.tile_size,
            "patch_size": self.patch_size,
            "model_dim": self.model_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "code_dim": self.code_dim,
            "num_embeddings": self.num_embeddings,
        }
