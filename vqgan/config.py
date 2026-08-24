from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    root: str = "images"
    out_dir: str = "data/processed"
    canvas_size: int = 1024
    min_short_side: int = 256
    downsample_factor: int = 8
    val_frac: float = 0.20
    seed: int = 0


@dataclass
class VQGANTrainConfig:
    data_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    out_dir: str = "outputs/vqgan"
    resume: str = ""  # path to a checkpoint saved by this script; "" = train from scratch

    latent_dim: int = 1024
    num_embeddings: int = 2048

    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 100
    lr: float = 1e-4

    ema_warmup_epochs: int = 2          # gradient-based codebook updates before switching to EMA
    disc_warmup_epochs: float = 1.0     # epochs before adversarial loss contributes to g_loss
    disc_weight: float = 0.5
    use_lpips: bool = True
    lpips_weight: float = 1.0

    amp: bool = True            # autocast + train in bf16 (no GradScaler needed for bf16)
    grad_clip_norm: float = 1.0
    log_every: int = 50
    eval_every_epochs: int = 1
    checkpoint_every_epochs: int = 5

    seed: int = 0
