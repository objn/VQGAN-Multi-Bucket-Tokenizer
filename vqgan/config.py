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
    # batch-bound, so a larger batch buys ~3% for 2.7GB and a spill risk.
    batch_size: int = 8
    # The loader streams ~1100 crops/s with 6 workers against a GPU that eats
    # ~30, so it is nowhere near the bottleneck; 4 keeps a 6x margin while
    # holding ~3GB less host RAM in parquet buffers.
    num_workers: int = 4
    # Every image contributes its *whole* tile grid, so there is no crops-per-image
    # cap; a big photo just yields more tiles. On ImageNet that averages ~4.9
    # tiles per usable image at overlap 0 (median 4, max seen 100).
    #
    # 0 = tiles butt up against each other. Raise it to mirror what
    # scripts/reconstruct.py does at inference (it defaults to 64px of
    # cross-faded overlap), at the cost of more redundant training crops:
    # overlap 64 yields ~7.2 tiles/image instead of ~4.9.
    tile_overlap: int = 0
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
    # model size, against ~11 minutes of training between evals, so the
    # in-training readout walks a deterministic prefix instead.
    eval_batches: int = 64

    # Schedule is in optimizer steps, not epochs: one pass over ImageNet-1k is
    # ~80k steps at this batch size, so "epoch" is too coarse a unit to
    # checkpoint, evaluate or warm up on.
    max_steps: int = 1_000_000
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
    ema_warmup_steps: int = 2_000
    disc_warmup_steps: int = 10_000   # steps before adversarial loss contributes to g_loss
    eval_every_steps: int = 2_000
    checkpoint_every_steps: int = 10_000
    log_every: int = 2_000

    lr: float = 1e-4
    min_lr: float = 1e-6

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
    disc_weight: float = 1.0
    use_lpips: bool = True

    amp: bool = True            # autocast + train in bf16 (no GradScaler needed for bf16)
    grad_clip_norm: float = 1.0

    seed: int = 24

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
