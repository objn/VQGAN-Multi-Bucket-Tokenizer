# VQGAN Multi-Resolution Bucket Tokenizer (max side 768)

A VQGAN (Esser et al. 2021 / Taming Transformers) image tokenizer for a larger AR
text-to-image pipeline. Unlike a fixed-square VQGAN, this one handles **multiple
aspect ratios** via aspect-ratio bucketing (the same technique SDXL and similar
systems use): every image is assigned to the closest bucket in the table below, where
the longest side is 768px (grid up to 24×24) and both sides are multiples of 32 (the
downsampling factor). This repo covers only the tokenizer — encoder, quantizer,
decoder, discriminator, losses, and training/eval. The AR transformer, the text
encoder, and the canvas/EMPTY-token system that reconciles sequence length across
buckets for the AR model are all out of scope.

| Ratio | Resolution (px) | Grid  | Tokens |
|-------|------------------|-------|--------|
| 1:1   | 768×768          | 24×24 | 576    |
| 4:5   | 608×768          | 19×24 | 456    |
| 5:4   | 768×608          | 24×19 | 456    |
| 4:3   | 768×576          | 24×18 | 432    |
| 3:4   | 576×768          | 18×24 | 432    |
| 3:2   | 768×512          | 24×16 | 384    |
| 2:3   | 512×768          | 16×24 | 384    |
| 16:9  | 768×448          | 24×14 | 336    |
| 9:16  | 448×768          | 14×24 | 336    |

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

The `_with_coco_mini` variants download+extract COCO train2017/val2017 (skipped if
already populated), then pass `--train-dir`/`--val-dir` to the underlying train
script pointing at wherever they extracted to. **They default to `$HOME/vqgan_data`,
not the repo's `data/` dir** — on RunPod the repo typically lives on a Network Volume
(commonly mounted at `/workspace`) for persistence, but writing 100k+ small `.jpg`
files there is slow; `$HOME` is normally the pod's local Container Disk instead (fast,
but wiped if the pod is terminated, not just stopped). Override with `VQGAN_DATA_DIR`
if your setup differs, e.g. to keep the dataset on the network volume anyway (slower
extraction, but survives pod termination):
```bash
VQGAN_DATA_DIR=data ./train_with_coco_mini.sh
```
The plain `train.sh`/`train_uv.sh` scripts assume the data is already extracted
somewhere and take `--train-dir`/`--val-dir` directly. All four forward extra args
straight to `vqgan.train`, e.g.
`./train_uv.sh --resume-from runs/vqgan-multi/checkpoints/latest.pt`.

## Data

Point `--train-dir` / `--val-dir` at a folder of images (any nesting, no captions or
metadata needed). Recommended starting datasets:

- **COCO minitrain** (~25K images) — for pipeline smoke-testing
- **COCO 2017 full** (~118K images) — for the actual training run
- Note: COCO images average ~640×480 / 480×640 (a 4:3 / 3:4-ish ratio). Mapped into
  this bucket system they land near the 4:3 / 3:4 bucket (768×576 / 576×768),
  roughly a 1.2× upscale from native resolution — mild, but still worth being
  aware of. Treat COCO as sufficient for
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

- **`batch_size`/`accumulation_steps`** (`train.batch_size`, `train.accumulation_steps`
  — also overridable via `--batch-size`/`--accumulation-steps`) are global, applied
  uniformly across every bucket: simpler to configure and reason about than a
  batch_size/accumulation_steps pair per bucket, at the cost of not maximizing VRAM
  usage on buckets smaller than the largest (1:1, 768×768) — fine at this project's
  scale. Size `batch_size` to fit the largest bucket without OOM, and
  `accumulation_steps` to reach whatever total images-per-update you want.
- **Bucketed batch sampler** (`BucketedBatchSampler` in `vqgan/data/buckets.py`):
  groups dataset indices by bucket, shuffles within each bucket, and shuffles batch
  order too so buckets interleave through an epoch — every yielded batch is
  single-bucket by construction, no padding/collating across shapes.
- bf16 mixed precision, gradient checkpointing on encoder/decoder
- Two-phase schedule: reconstruction + codebook + perceptual loss only for the first
  `discriminator_start_step` steps (default 30,000), then the PatchGAN discriminator
  and adversarial loss activate
- Separate AdamW optimizers for generator (encoder+decoder+quantizer) and
  discriminator, with independently configurable learning rates
- Checkpoints (model + both optimizers + both LR schedulers + step/epoch/bucket-sampler
  batch offset + torch/numpy/python RNG state) saved every `--checkpoint-every` steps
  to `<output-dir>/checkpoints/`, so training resumes mid-epoch across sessions with
  the bucket sampler's batch order (and data augmentation stream) reproducible across
  resumes — resume always lands on an accumulation boundary, never mid-accumulation.
  Writes are atomic (temp file + rename), so a crash or Ctrl+C mid-save never leaves a
  corrupt `latest.pt` behind. Resuming with a checkpoint whose architecture-critical
  config (codebook size, channel dims, etc.) doesn't match the current config fails
  loudly with a clear diff, instead of silently corrupting training or crashing deep
  inside `load_state_dict`.
- **Resume is the default.** Re-running the same command auto-detects
  `<output-dir>/checkpoints/latest.pt` and resumes from it — no need to pass
  `--resume-from` manually. Pass `--fresh` to start over from step 0 instead, or
  `--resume-from <path>` to resume from a specific (non-latest) checkpoint:

```bash
uv run python -m vqgan.train --resume-from runs/vqgan-multi/checkpoints/some_older_step.pt ...
uv run python -m vqgan.train --fresh ...
```

- **Checkpoint retention**: a rolling window of the most recent `keep_last_n_checkpoints`
  (default 5) `step_*.pt` files is kept; older ones are deleted automatically, except
  those landing on a permanent milestone (`step % checkpoint_milestone_every == 0`,
  default every 10,000 steps), which are kept forever. `latest.pt` is never pruned.
- **Ctrl+C is safe**: `KeyboardInterrupt` is caught and triggers one best-effort final
  checkpoint save to `latest.pt` before exiting, so a manual stop doesn't lose progress
  since the last periodic save.

### Training length is epoch-based, not a raw step count

`--total-steps` doesn't exist as a CLI flag or a value you set. The only training-length
knob is **`target_epochs`** (`train.target_epochs`, default 30) — 1 epoch = the model
has seen every image in the training dataset once, regardless of `batch_size`.
`total_steps` is *derived* fresh at the start of every run:

```
total_steps = target_epochs * steps_per_epoch
```

`steps_per_epoch` is computed **per bucket** (a bucket's remainder images that don't
fill a full batch are dropped, since a batch must be single-bucket) and summed, since
bucket dataset sizes differ even though `batch_size`/`accumulation_steps` are global:

```
steps_per_epoch = (sum over buckets of: images in that bucket // batch_size) // accumulation_steps
```

— see `steps_per_epoch()` in `vqgan/data/buckets.py`. This is recomputed from the
*actual* dataset every run (never hardcoded/cached), so moving from COCO minitrain
(~25K images) to COCO full (~118K images) with the same `target_epochs` automatically
trains for proportionally more steps — no manual recalculation needed. Changing
`batch_size`/`accumulation_steps` changes how many steps make up an epoch, but
`target_epochs` itself doesn't need to change — the model still sees the same amount of
data per epoch either way.

`total_steps` (and the resolved `steps_per_epoch`) are printed at the start of every
run and written back into `configs/vqgan-multi.json` (`train.total_steps`) for
auditability — see "Config File Sync" below. It's informational/self-describing only,
never read back in as an input.

### Config File Sync

`configs/vqgan-multi.json` always reflects the hyperparameters actually in use for the
most recent run, not just whatever was on disk at launch. Any CLI-overridden value, and
the freshly-derived `total_steps`, gets written back atomically (temp file + rename) at
startup if it differs from what's on disk — with a console log line
(`Config file updated: ...`) whenever this happens, so it's never a silent change.
`resume_from` is excluded from this sync (it's transient per-invocation state — an
auto-detected `latest.pt` path isn't a hyperparameter worth persisting).

### Codebook health (avoiding collapse)

A collapsed codebook (perplexity stuck near 1-2 out of `codebook_size`, i.e. only a
handful of codes ever get selected) is a common VQ-VAE/VQGAN failure mode. This repo
has several mitigations, all config-driven and on by default in `configs/vqgan-multi.json`:

- **EMA codebook updates** (`model.use_ema`) — more stable than gradient-based codebook
  updates; see `VectorQuantizer._update_ema()`.
- **Lower commitment β** (`model.commitment_beta`, default 0.15) — too strong a
  commitment term over-constrains the encoder early in training.
- **Dead-code revival** (`model.dead_code_revival`, `train.dead_code_revival_every`) —
  every N effective steps, codebook entries with zero accumulated usage since the last
  check are reset to random encoder-output vectors from the current batch (and their
  EMA stats reseeded, if `use_ema` is on), giving them a chance to start attracting
  nearest-neighbor traffic instead of sitting dead forever.
- **K-means codebook init** (`model.codebook_init: "kmeans"`) — replaces the default
  tiny-range uniform init (`±1/codebook_size`, which can be far from the encoder's
  actual output scale) with centroids computed from real encoder output over
  `train.kmeans_init_batches` initial batches, run once before training starts. Falls
  back to uniform init with a console warning if fewer than `codebook_size` vectors
  were collected (increase `kmeans_init_batches` or `batch_size` if this fires
  unexpectedly).
- **LR warmup** (`train.lr_warmup_steps`) — linear warmup for `opt_g` from the start of
  training; `opt_d` gets its own independent warmup window starting whenever the
  discriminator activates (`loss.discriminator_start_step`), automatically, with no
  extra config needed.

**Known limitation**: dead-code revival and k-means init both write directly to
`embedding.weight.data` outside the optimizer/gradient-sync path. Under multi-GPU DDP,
each rank runs these independently from its own local batch data and the result is
**not broadcast across ranks** — the codebook can silently diverge between GPUs over a
long run. A one-time console warning fires on rank 0 if this situation is hit; fixing
it (broadcasting rank 0's result to all ranks) is a follow-up, not yet implemented.

### Multi-GPU (DistributedDataParallel)

Set `VQGAN_NUM_GPUS` when using `train.sh`/`train_uv.sh` (or `train_with_coco_mini*.sh`,
which forward it through) to train across multiple GPUs on one machine via
`torchrun`, one process per GPU:

```bash
VQGAN_NUM_GPUS=2 ./train_uv.sh --train-dir data/train --val-dir data/val
```

Each GPU gets a disjoint shard of the same deterministic per-epoch batch order (every
rank computes the identical shuffle from the same seed, then takes a `[rank::world_size]`
slice — see `BucketedBatchSampler` in `vqgan/data/buckets.py`), so every rank performs
exactly the same number of weight updates per epoch (required since DDP's backward pass
is a collective operation) and there's no data overlap or duplication across GPUs.
`--batch-size` is **per GPU**, not total; `steps_per_epoch`/`total_steps` are computed
with `world_size` folded in, so adding GPUs alone increases effective throughput per
epoch without changing `target_epochs`. Only rank 0 writes TensorBoard logs, prints
progress, and saves checkpoints; checkpoints are saved with clean (non-DDP-prefixed)
keys, so they load identically whether you resume on 1 GPU or N GPUs. `--num-workers`
is still per-process, so with `VQGAN_NUM_GPUS=2` you get `2 × num_workers` total
DataLoader worker processes — lower `--num-workers` if that oversubscribes the pod's
CPU count.

This does **not** by itself fix an out-of-memory error — each GPU still needs to fit its
own `--batch-size` independently. If you're hitting OOM, lower `--batch-size` (raise
`--accumulation-steps` to compensate if you want images-per-update to stay roughly the
same), then add `VQGAN_NUM_GPUS` for throughput.

### Monitoring

All metrics (reconstruction/codebook/perceptual/adversarial/discriminator loss,
codebook perplexity and usage, learning rates) are logged to TensorBoard under
`train/*`, **plus a per-bucket breakdown** under `train_bucket_<name>/*` — quality can
differ meaningfully across buckets and this needs to stay visible instead of getting
averaged away. `train/codebook_perplexity`, `train/codebook_usage`, and
`train/codebook_num_revived` are logged **every step** (not gated by `--log-every`)
since they're exactly the kind of bursty, step-local signal you want at full resolution
when diagnosing codebook collapse — other losses stay on the `--log-every` cadence.
`train/epoch` is also logged every step, so TensorBoard's x-axis can be cross-referenced
against epoch boundaries. Periodic input-vs-reconstruction image grids are logged per
bucket too (`train_bucket_<name>/input_vs_reconstruction`), so all aspect ratios get
visually sanity-checked as training cycles through buckets, not just whichever one
happened to log last. **VRAM headroom** is logged every step as
`system/vram_allocated_gb` and `system/vram_reserved_gb`
(`torch.cuda.memory_allocated()` / `torch.cuda.memory_reserved()`), so it's visible over
time in the dashboard rather than only discoverable after a slowdown or OOM:

```bash
tensorboard --logdir runs/vqgan-multi/tensorboard
```

### Progress reporting

The progress bar shows **both epoch progress and step progress**, since epochs are the
primary training-length unit and shouldn't be buried behind a raw step counter:

```
Epoch 3/30 | step 4021/7615 (this epoch): 412000/4500000 [elapsed, s/step=1.05, ETA=..., recon=..., ...]
```

`n_fmt/total_fmt` (`412000/4500000` above) tracks the persistent **overall** step count
(`step`/`total_steps`) across resumes; the description prefix tracks the current epoch
(`Epoch 3/30`) and progress **within that epoch** (`step 4021/7615`, i.e.
`step_in_epoch`/`steps_per_epoch`). Both are whole steps (post-accumulation) — never
fractional/micro-batch counts. Step speed (`s/step`) and `ETA` in the postfix use a
rolling average over the last 50 steps rather than the most recent step alone, since the
first few steps include CUDA warmup / cuDNN autotune overhead that would otherwise skew
a single-step estimate badly.

At the start of every run (and on resume), `batch_size`/`accumulation_steps`,
`target_epochs`, and the derived `steps_per_epoch`/`total_steps` are printed to the
console so past runs are auditable from the log alone:

```
world_size=1 batch_size=2 accumulation_steps=8 target_epochs=30 steps_per_epoch=1667 total_steps=50010
```

If a step's measured duration exceeds 5× the rolling average, a console warning is
printed — this is the signature of VRAM spillover into shared/system memory (via
PCIe), not a normal slowdown, and is surfaced immediately rather than silently
absorbed into the rolling average.

## Tests

```bash
uv sync --group dev
uv run pytest tests/
```

`tests/test_shape_agnostic.py` runs a forward+backward pass through two different
bucket aspect ratios on the same tiny model instance and asserts every parameter gets
a gradient at both shapes — this is the explicit check that the encoder/decoder/
quantizer/discriminator make no hardcoded-shape assumptions anywhere, which is what
lets one set of weights handle every bucket in the table.

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
- EMA codebook updates (`--use-ema`) are **on by default** in `configs/vqgan-multi.json`
  as a more stable alternative to gradient-based codebook updates.
- Dead-code revival (`--dead-code-revival`) periodically resets unused codebook entries
  to random encoder-output vectors — see "Codebook health" above.
- K-means codebook init (`--codebook-init kmeans`) seeds the codebook from real encoder
  output instead of a tiny-range uniform init — see "Codebook health" above.
- LR warmup (`--lr-warmup-steps`) ramps the generator's (and, independently, the
  discriminator's) learning rate up linearly at the start of training/activation.
- This VQGAN's job ends at: given an image, assign it to the correct bucket and
  produce that bucket's token sequence; given a token sequence and its bucket,
  reconstruct the image. Making sequence length uniform across buckets for the AR
  model (the canvas/EMPTY-token system) is a separate, later task.
