"""Central hyperparameter configuration for VQGAN training/eval.

The single source of truth for hyperparameter values is a JSON file
(default: configs/vqgan-multi.json), loaded via `load_config()`. These dataclasses
define the schema/types and serve as the fallback when no JSON file is given.
"""
import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from vqgan.data.buckets import DEFAULT_BUCKETS


@dataclass
class ModelConfig:
    in_channels: int = 3
    base_channels: int = 128
    channel_mults: tuple = (1, 1, 2, 2, 4)  # 5 stages -> 32x downsampling
    latent_channels: int = 256
    num_res_blocks: int = 2

    codebook_size: int = 16384
    codebook_dim: int = 256  # must match latent_channels
    commitment_beta: float = 0.15
    use_ema: bool = False
    ema_decay: float = 0.99
    dead_code_revival: bool = False
    codebook_init: str = "uniform"  # "uniform" or "kmeans" (see run_kmeans_init() in train.py)

    discriminator_channels: int = 64
    discriminator_num_layers: int = 3

    def __post_init__(self) -> None:
        self.channel_mults = tuple(self.channel_mults)  # JSON has no tuple type; normalize on load


@dataclass
class LossConfig:
    lambda_perceptual: float = 1.0
    lambda_adv: float = 0.1
    adv_loss_type: str = "hinge"  # "hinge" or "non_saturating"
    discriminator_start_step: int = 30000


@dataclass
class DataConfig:
    train_dir: str = "data/train"
    val_dir: str = "data/val"
    # aspect-ratio bucket table: list of {"name", "width", "height"} (both multiples of
    # 32, longest side 768px). See vqgan/data/buckets.py for the Bucket type and
    # assignment logic. batch_size/accumulation_steps are global (TrainConfig below),
    # not per-bucket -- simpler to configure and tune than one pair per bucket.
    buckets: list = field(
        default_factory=lambda: [dataclasses.asdict(b) for b in DEFAULT_BUCKETS]
    )
    horizontal_flip: bool = True
    num_workers: int = 8


@dataclass
class TrainConfig:
    output_dir: str = "runs/vqgan-multi"

    # The only training-length knob the user sets. 1 epoch = the model has seen every
    # image in the training dataset once, regardless of batch_size (see
    # .claude/mulit-vqgan.md, "Training Length -- Epoch-Based"). total_steps is NOT
    # configured directly -- it's derived at startup as target_epochs *
    # steps_per_epoch(...), recomputed every run from the actual dataset size (see
    # steps_per_epoch() in vqgan/data/buckets.py).
    target_epochs: int = 30

    # informational/self-describing only: the total_steps actually derived and used for
    # the most recent run, written back here at startup for auditability (see
    # .claude/mulit-vqgan.md, "Config File Sync"). Never read as an input -- always
    # recomputed from target_epochs + the live dataset/batch settings. Absent from a
    # freshly-written config until the first run computes it.
    total_steps: int = 0

    # global (not per-bucket): simpler to configure/tune than a batch_size +
    # accumulation_steps pair per bucket, at the cost of not maximizing VRAM usage on
    # buckets smaller than the largest (1:1, 768x768) -- fine for this project's scale.
    batch_size: int = 2
    accumulation_steps: int = 8

    lr_generator: float = 4.5e-6
    lr_discriminator: float = 4.5e-6
    beta1: float = 0.5
    beta2: float = 0.9
    lr_warmup_steps: int = 1500  # linear LR warmup, in effective steps; opt_d gets its own
    # independent warmup window starting whenever the discriminator activates (see train.py)

    mixed_precision: bool = True  # bf16
    grad_checkpointing: bool = True

    log_every: int = 50
    image_log_every: int = 500
    checkpoint_every: int = 2000
    val_every: int = 2000
    keep_last_n_checkpoints: int = 5  # rolling window of recent non-milestone checkpoints to retain
    checkpoint_milestone_every: int = 10000  # kept forever regardless of keep_last_n_checkpoints
    dead_code_revival_every: int = 400  # effective steps between dead-code revival checks

    # k-means codebook init (only used when model.codebook_init == "kmeans"): number of physical
    # batches of encoder output to collect before running k-means, and Lloyd's-algorithm iteration
    # count. kmeans_init_batches must be large enough that batches * tokens_per_image >= codebook_size
    # for the smallest bucket, or k-means falls back to the uniform init with a warning.
    kmeans_init_batches: int = 100
    kmeans_init_iters: int = 15

    num_workers: int = 8
    seed: int = 42
    resume_from: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            model=ModelConfig(**d.get("model", {})),
            loss=LossConfig(**d.get("loss", {})),
            data=DataConfig(**d.get("data", {})),
            train=TrainConfig(**d.get("train", {})),
        )

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        """Atomic write (temp file + rename) -- a crash or Ctrl+C mid-write must never
        leave configs/vqgan-multi.json half-written (see .claude/mulit-vqgan.md,
        "Config File Sync")."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.with_suffix(p.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp_path, p)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "vqgan-multi.json"


def load_config(path: str | None = None) -> Config:
    """Loads the single-source-of-truth JSON config. Falls back to dataclass
    defaults (and DEFAULT_CONFIG_PATH if it exists) when no path is given."""
    if path:
        return Config.from_json(path)
    if DEFAULT_CONFIG_PATH.exists():
        return Config.from_json(str(DEFAULT_CONFIG_PATH))
    return Config()
