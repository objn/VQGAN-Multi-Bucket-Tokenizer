# VQGAN Multi-Resolution Bucket Tokenizer (max side 1024)

A VQGAN (Esser et al. 2021 / Taming Transformers) image tokenizer for a larger AR
text-to-image pipeline. Unlike a fixed-square VQGAN, this one handles **multiple
aspect ratios** via aspect-ratio bucketing (the same technique SDXL and similar
systems use): every image is assigned to the closest bucket in the table below, where
the longest side is 1024px and both sides are multiples of 32 (the downsampling
factor). This repo covers only the tokenizer — encoder, quantizer, decoder,
discriminator, losses, and training/eval. The AR transformer, the text encoder, and
the canvas/EMPTY-token system that reconciles sequence length across buckets for the
AR model are all out of scope.

| Ratio | Resolution (px) | Grid  | Tokens |
|-------|------------------|-------|--------|
| 1:1   | 1024×1024        | 32×32 | 1024   |
| 4:5   | 832×1024         | 26×32 | 832    |
| 5:4   | 1024×832         | 32×26 | 832    |
| 4:3   | 1024×768         | 32×24 | 768    |
| 3:4   | 768×1024         | 24×32 | 768    |
| 3:2   | 1024×672         | 32×21 | 672    |
| 2:3   | 672×1024         | 21×32 | 672    |
| 16:9  | 1024×576         | 32×18 | 576    |
| 9:16  | 576×1024         | 18×32 | 576    |

Every training batch contains images from a **single bucket only** — token count
differs per bucket, so mixed-bucket batches can't collate into one tensor.

Designed to fit a single RTX 3080 Ti (12GB VRAM): bf16 mixed precision, gradient
checkpointing, and gradient accumulation are on by default.

## Why this works with one set of weights

The encoder, decoder, and discriminator are **fully convolutional with no fixed-size
assumptions** — no `nn.Linear`/`nn.Flatten` anywhere that hardcodes a spatial
dimension, no hardcoded H/W in any `forward()`. The quantizer is per-token and
shape-agnostic by construction (it flattens `[B, C, H, W]` to `[B*H*W, C]` for the
codebook lookup). The attention blocks at the encoder/decoder bottleneck use a 2D
sinusoidal position embedding computed on the fly from the actual input H/W (see
`build_2d_sincos_position_embedding` in `vqgan/models/common.py`) instead of a
fixed-size positional embedding table, so the same attention weights generalize across
every bucket's grid shape. LPIPS/VGG is likewise fully convolutional and accepts
variable input resolutions natively.

This was verified directly: the same model weights run a forward+backward pass across
multiple different buckets (1:1, 16:9, 3:4) in the same test with no shape assumptions
leaking in anywhere.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for environment/package management.

```bash
uv sync
```

### Quick-start scripts

- **`train.sh`** / **`train_with_coco_mini.sh`** — plain `python`, no uv/.venv. For
  environments that already ship torch/torchvision at the system level, e.g. RunPod's
  PyTorch cluster templates (install the remaining deps first: `pip install -r
  requirements.txt` — safe even with torch preinstalled, pip skips already-satisfied
  packages).
- **`train_uv.sh`** / **`train_with_coco_mini_uv.sh`** — uv-managed equivalents (runs
  `uv sync` implicitly via `uv run`), for local dev machines without a pre-installed
  torch.

The `_with_coco_mini` variants download+extract COCO train2017/val2017 into
`data/train`/`data/val` (skipped if already populated) before training; the plain
variants assume the data is already there. All four forward extra args straight to
`vqgan.train`, e.g. `./train_uv.sh --resume-from runs/vqgan-multi/checkpoints/latest.pt`.

## Data

Point `--train-dir` / `--val-dir` at a folder of images (any nesting, no captions or
metadata needed). Recommended starting datasets:

- **COCO minitrain** (~25K images) — for pipeline smoke-testing
- **COCO 2017 full** (~118K images) — for the actual training run
- Note: COCO images average ~640×480 / 480×640 (a 4:3 / 3:4-ish ratio). Mapped into
  this bucket system they land near the 4:3 / 3:4 bucket (1024×768 / 768×1024),
  roughly a 1.6× upscale from native resolution — a real quality ceiling worth being
  aware of, no matter how well the VQGAN trains. Treat COCO as sufficient for
  validating that the pipeline and bucket system work correctly; consider a
  higher-resolution dataset (e.g. Unsplash Lite/Full) before treating final
  reconstruction quality numbers as representative.
- COCO's native aspect ratios cluster around 4:3/3:4 — it will not naturally populate
  the 1:1, 16:9, 9:16, or other buckets well. If bucket coverage matters for a given
  run, supplement with a source that has a wider natural spread of aspect ratios.

**Bucket assignment**: each image's aspect ratio is compared (log-ratio distance) to
every bucket in the table and assigned to the closest one, computed once per image at
dataset construction (`assign_bucket()` in `vqgan/data/buckets.py`, header-only PIL
reads so this is cheap even over 100k+ images). Each image is then resized preserving
aspect ratio ("cover" fit, never stretched/squashed) and cropped to its bucket's exact
resolution — random crop + random horizontal flip for training, center crop for
validation. No color augmentation, since the model needs to learn accurate color
reproduction.

## Configuration

All hyperparameters — including the bucket table — live in a single JSON file,
**`configs/vqgan-multi.json`** — this is the single source of truth for
model/loss/data/train settings. `vqgan/config.py` dataclasses (`ModelConfig`,
`LossConfig`, `DataConfig`, `TrainConfig`) just define the schema/types and are the
fallback if no JSON file is found.

```bash
uv run python -m vqgan.train --config configs/vqgan-multi.json
```

`--config` defaults to `configs/vqgan-multi.json`, so plain
`uv run python -m vqgan.train` picks it up automatically. Every CLI flag below is an
**optional override** on top of whatever the JSON file specifies — only flags you
actually pass take effect; anything you omit keeps its JSON value. To customize the
bucket table itself (add/remove buckets, change resolutions), edit the `data.buckets`
array in the JSON file directly — there's no CLI flag for it, since it's structured
data, not a scalar.

## Training

```bash
uv run python -m vqgan.train \
  --train-dir data/train \
  --val-dir data/val \
  --output-dir runs/vqgan-multi
```

Key defaults (from `configs/vqgan-multi.json`), all overridable via CLI flags (see
`vqgan/train.py --help`):

- Uniform physical batch size 3 across all buckets (sized to fit the largest, 1:1 at
  1024×1024 — smaller buckets use less VRAM at the same batch size, no per-bucket
  tuning by default), grad accumulation ×5 (effective batch size 15)
- **Bucketed batch sampler** (`BucketedBatchSampler` in `vqgan/data/buckets.py`):
  groups dataset indices by bucket, shuffles within each bucket, and shuffles bucket
  order too so buckets interleave through an epoch — every yielded batch is
  single-bucket by construction, no padding/collating across shapes
- bf16 mixed precision, gradient checkpointing on encoder/decoder
- Two-phase schedule: reconstruction + codebook + perceptual loss only for the first
  `discriminator_start_step` steps (default 30,000), then the PatchGAN discriminator
  and adversarial loss activate
- Separate AdamW optimizers for generator (encoder+decoder+quantizer) and
  discriminator, with independently configurable learning rates
- Checkpoints (model + both optimizers + step/epoch/bucket-sampler batch offset) saved
  every `--checkpoint-every` steps to `<output-dir>/checkpoints/`, so training can
  resume mid-epoch across sessions, with the bucket sampler's batch order
  deterministic/reproducible across resumes:

```bash
uv run python -m vqgan.train --resume-from runs/vqgan-multi/checkpoints/latest.pt ...
```

### Monitoring

All metrics (reconstruction/codebook/perceptual/adversarial/discriminator loss,
codebook perplexity and usage, learning rates) are logged to TensorBoard under
`train/*`, **plus a per-bucket breakdown** under `train_bucket_<name>/*` — quality can
differ meaningfully across buckets and this needs to stay visible instead of getting
averaged away. Periodic input-vs-reconstruction image grids are logged per bucket too
(`train_bucket_<name>/input_vs_reconstruction`), so all aspect ratios get visually
sanity-checked as training cycles through buckets, not just whichever one happened to
log last. **VRAM headroom** is logged every effective step as `system/vram_allocated_gb`
and `system/vram_reserved_gb` (`torch.cuda.memory_allocated()` /
`torch.cuda.memory_reserved()`), so it's visible over time in the dashboard rather than
only discoverable after a slowdown or OOM:

```bash
tensorboard --logdir runs/vqgan-multi/tensorboard
```

### Progress reporting

The progress bar counts **effective steps only** (post-gradient-accumulation) as
integers — `current_effective_step / total_effective_steps` — never fractional
physical sub-steps. Step speed (`s/step`) and `ETA` shown in the bar's postfix use a
rolling average over the last 50 effective steps rather than the most recent step
alone, since the first few steps include CUDA warmup / cuDNN autotune overhead that
would otherwise skew a single-step estimate badly.

`total_effective_steps` (`--total-steps`) is a fixed target independent of batch size:
it represents a fixed amount of data the model should see, not a fixed number of
physical batches. If you change `--grad-accum-steps` later (e.g. to fit VRAM after
lowering `--batch-size`), `--total-steps` does **not** need to change — only
`physical_batch_size × grad_accum_steps` (the effective batch size) needs to stay
constant to keep training comparable.

At the start of every run (and on resume), the physical batch size, accumulation
steps, and resulting effective batch size are printed to the console so past runs are
auditable from the log alone:

```
physical_batch_size=3 accumulation_steps=5 effective_batch_size=15 total_effective_steps=300000
```

If a step's measured duration exceeds 5× the rolling average, a console warning is
printed — this is the signature of VRAM spillover into shared/system memory (via
PCIe), not a normal slowdown, and is surfaced immediately rather than silently
absorbed into the rolling average.

## Evaluation

Standalone script, separate from training, run against a checkpoint and a held-out
validation split:

```bash
uv run python -m vqgan.eval --checkpoint runs/vqgan-multi/checkpoints/latest.pt --val-dir data/val
```

Reports both **overall** and **per-bucket** breakdowns (a bucket with less training
data, e.g. extreme ratios like 16:9, may lag behind the more common ones — this needs
to be visible, not averaged away):

- **rFID** (reconstruction FID, via Inception v3 features) — primary quality metric,
  target < 2.0 overall; per-bucket rFID is skipped (reported as `n/a`) for buckets with
  too few validation images for a stable covariance estimate
- **LPIPS** — average over the val set, overall and per-bucket
- **PSNR** — secondary; don't over-optimize for it, high PSNR can coexist with blurry output
- **Codebook utilization** — % of codebook entries used across the whole val set; warns
  if below 50%, since low utilization means wasted codebook capacity

## Package layout

```
configs/
  vqgan-multi.json  single source of truth for all hyperparameters (model/loss/data/train/buckets)
vqgan/
  models/           encoder, decoder, quantizer (VQ + optional EMA), PatchGAN discriminator, VQGAN wrapper
                     -- all fully convolutional / shape-agnostic, no per-resolution construction args
  losses/           reconstruction (L1), LPIPS perceptual, adversarial (hinge/non-saturating), combined schedule
  data/
    buckets.py      bucket table, aspect-ratio assignment, BucketedBatchSampler
    dataset.py      bucketed image-folder dataset (resize-to-cover + crop per assigned bucket)
  train.py          training entrypoint
  eval.py           standalone evaluation (overall + per-bucket rFID / LPIPS / PSNR / codebook utilization)
  config.py         dataclass schema + load_config()/Config.from_json() for reading configs/*.json
```

## Notes

- Decoder output range is `[-1, 1]` (`tanh`); the data pipeline normalizes inputs to
  match (`mean=0.5, std=0.5` per channel).
- Codebook size (default 16,384) and embedding dim are constructor args on
  `VectorQuantizer`, so scaling to 65,536 later doesn't require rewriting the class.
- EMA codebook updates are available via `--use-ema` as a more stable alternative to
  gradient-based codebook updates.
- This VQGAN's job ends at: given an image, assign it to the correct bucket and
  produce that bucket's token sequence; given a token sequence and its bucket,
  reconstruct the image. Making sequence length uniform across buckets for the AR
  model (the canvas/EMPTY-token system) is a separate, later task.
