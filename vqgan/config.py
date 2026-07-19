"""Central hyperparameter configuration for VQGAN training/eval.

The single source of truth for hyperparameter values is a JSON file
(default: configs/vqgan-multi.json), loaded via `load_config()`. These dataclasses
define the schema/types and serve as the fallback when no JSON file is given.
"""
import dataclasses
import json
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
    commitment_beta: float = 0.25
    use_ema: bool = False
    ema_decay: float = 0.99

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
    # aspect-ratio bucket table: list of {"name", "width", "height"} (both multiples of 32,
    # longest side 1024px). See vqgan/data/buckets.py for the Bucket type and assignment logic.
    buckets: list = field(
        default_factory=lambda: [dataclasses.asdict(b) for b in DEFAULT_BUCKETS]
    )
    horizontal_flip: bool = True
    num_workers: int = 8


@dataclass
class TrainConfig:
    output_dir: str = "runs/vqgan-multi"
    total_steps: int = 300000

    # uniform physical batch size across all buckets, sized to fit the largest (1:1, 1024x1024);
    # smaller buckets use less VRAM at the same batch size (no per-bucket tuning by default)
    batch_size: int = 3
    grad_accum_steps: int = 5  # effective batch size = batch_size * grad_accum_steps

    lr_generator: float = 4.5e-6
    lr_discriminator: float = 4.5e-6
    beta1: float = 0.5
    beta2: float = 0.9

    mixed_precision: bool = True  # bf16
    grad_checkpointing: bool = True

    log_every: int = 50
    image_log_every: int = 500
    checkpoint_every: int = 2000
    val_every: int = 2000

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
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "vqgan-multi.json"


def load_config(path: str | None = None) -> Config:
    """Loads the single-source-of-truth JSON config. Falls back to dataclass
    defaults (and DEFAULT_CONFIG_PATH if it exists) when no path is given."""
    if path:
        return Config.from_json(path)
    if DEFAULT_CONFIG_PATH.exists():
        return Config.from_json(str(DEFAULT_CONFIG_PATH))
    return Config()
