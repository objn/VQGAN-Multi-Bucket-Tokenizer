# VQGAN Implementation Spec — Multi-Resolution Bucket Tokenizer (max side 768)

## Context

This VQGAN is the image tokenizer for a larger AR text-to-image pipeline (Qwen3.5 text backbone + custom AR transformer, not part of this task). This task covers **only the VQGAN tokenizer**: encoder, quantizer, decoder, discriminator, losses, and training loop. Build it as a clean, importable Python package, not a notebook.

Target hardware: single NVIDIA RTX 3080 Ti, 12GB VRAM, 64GB system RAM (DDR4). Design every default (batch size, precision, checkpointing) around fitting this GPU.

## Architecture Requirements

### Resolution / bucket system

This VQGAN must handle **multiple aspect ratios**, not a single fixed square size. The constraint is: **longest side = 768px**, with the other side determined by aspect ratio, both sides rounded to the nearest multiple of 32 (the downsampling factor). (Reduced from an earlier 1024px max-side target — the larger version trained too slowly and was overkill for the current dataset scale; revisit 1024 later once the pipeline and dataset are scaled up.)

Downsampling factor: **32×** (5 downsampling stages of ÷2 each: 2^5 = 32). This is fixed regardless of bucket.

Bucket table (compute more at runtime as needed, but these are the reference set to support first):

| Ratio | Resolution (px) | Grid   | Tokens |
|-------|------------------|--------|--------|
| 1:1   | 768×768          | 24×24  | 576    |
| 4:5   | 608×768          | 19×24  | 456    |
| 5:4   | 768×608          | 24×19  | 456    |
| 4:3   | 768×576          | 24×18  | 432    |
| 3:4   | 576×768          | 18×24  | 432    |
| 3:2   | 768×512          | 24×16  | 384    |
| 2:3   | 512×768          | 16×24  | 384    |
| 16:9  | 768×448          | 24×14  | 336    |
| 9:16  | 448×768          | 14×24  | 336    |

Because token count differs per bucket, **each training batch must contain images from a single bucket only** (see Data Pipeline below) — this is standard "aspect ratio bucketing" as used in SDXL and similar systems.

- Encoder and decoder must be **fully convolutional with no fixed-size assumptions** (no `nn.Linear` or `nn.Flatten` anywhere that hardcodes a spatial dimension, no hardcoded H/W in `forward()`). This is what lets the same encoder/decoder weights handle every bucket in the table without any architecture changes.
- Attention blocks (encoder bottleneck, decoder bottleneck) must work with variable sequence length — standard self-attention already does this, just don't hardcode a sequence length anywhere (e.g. no fixed positional embedding table sized for one specific grid — use a form of positional encoding that generalizes across grid shapes, such as 2D RoPE or a resolution-agnostic scheme).
- Encoder and decoder must be symmetric (mirrored architecture) at every bucket.

### Encoder
- CNN-based (ResNet-style residual blocks + downsampling convs), following the VQGAN (Esser et al. 2021) / Taming Transformers design
- 5 downsampling stages, each halving spatial resolution (e.g. for the 1:1 bucket: 768 → 384 → 192 → 96 → 48 → 24)
- Attention block(s) at the lowest resolution for global coherence
- Output: continuous feature map `[B, C, H/32, W/32]` (C = latent channel dim, e.g. 256) — H and W vary by bucket

### Quantizer (Vector Quantization)
- Codebook size: **16,384 entries** (2^14), configurable — expose as a constructor arg so it can be scaled to 65,536 later without rewriting the class
- Codebook embedding dim must match encoder output channel dim C
- Implement standard VQ-VAE quantization with straight-through gradient estimator
- Include **commitment loss** term (β, default 0.25) as part of the quantizer's forward return, not computed externally
- Quantization is per-token and shape-agnostic by construction — confirm no part of this module assumes a fixed grid size
- Track codebook usage (perplexity / active codes) and expose it in the forward output for logging — dead codebook entries are a common failure mode and must be observable during training
- Optional: implement EMA (exponential moving average) codebook updates instead of gradient-based updates (more stable in practice) — make this a config flag

### Decoder
- Mirrors the encoder: 5 upsampling stages back to the bucket's native resolution
- Residual blocks + attention at lowest resolution, symmetric to encoder
- Output: `[B, 3, H, W]` matching the input bucket's resolution, values in a defined range (document whether it's [-1,1] or [0,1] and be consistent with the data pipeline's normalization)

### Discriminator (for adversarial loss)
- PatchGAN-style discriminator (operates on image patches, not whole-image classification) — this is standard for VQGAN and much more stable than a global discriminator
- Must also be fully convolutional / shape-agnostic for the same reason as encoder/decoder — it needs to score patches from every bucket resolution
- Separate from the encoder/decoder, own optimizer

## Loss Function

Total generator loss is a weighted sum of four terms. Implement each as a separate, independently testable function:

1. **Reconstruction loss** — L1 (not L2) between input and reconstructed image
2. **Codebook / commitment loss** — returned from the quantizer forward pass (see above)
3. **Perceptual loss (LPIPS)** — use a pretrained LPIPS network (e.g. `lpips` package, VGG backbone) comparing input vs reconstruction. This is essential — L1 alone produces blurry results. Confirm the LPIPS backbone accepts variable input resolutions (standard LPIPS/VGG does, since it's also fully convolutional).
4. **Adversarial loss** — non-saturating GAN loss from the PatchGAN discriminator, applied to the generator (encoder+decoder+quantizer) output

```
L_total = L_recon + L_codebook + λ_perceptual * L_perceptual + λ_adv * L_adversarial
```

Make all λ weights configurable, not hardcoded.

### Discriminator loss
Standard hinge loss or non-saturating GAN loss, computed separately with its own optimizer step (alternating G/D updates per training step, standard GAN training pattern).

### Two-phase training schedule
- **Phase A (warmup):** train encoder+decoder+quantizer with only reconstruction + codebook + perceptual loss. No discriminator, no adversarial loss. Duration: first 20,000–50,000 steps (make this a config value, e.g. `discriminator_start_step`).
- **Phase B:** after `discriminator_start_step`, activate the discriminator and add the adversarial loss term to `L_total`. Do not start the discriminator from step 0 — this destabilizes early training.

## Tooling

- **Package/environment management: `uv`** — the repo should be set up for `uv` (pyproject.toml + uv.lock), not raw pip/venv or conda. `README.md` setup instructions should use `uv sync` / `uv run` commands.
- **Framework: PyTorch**
- **Experiment tracking: TensorBoard** — log all training-loop metrics there (see below), not to stdout only. Use `torch.utils.tensorboard.SummaryWriter`. README should include the `tensorboard --logdir ...` command to view it.

## Training Loop Requirements

- Mixed precision (bf16) — required to fit 768px-class training on 12GB VRAM
- Gradient checkpointing on encoder/decoder — make it a toggleable flag
- **Gradient accumulation** (see below)
- **Bucketed batch sampler**: a custom `Sampler` (or dataset wrapper) that groups images into their nearest bucket (by aspect ratio) and only yields batches where every image belongs to the same bucket. This is the mechanism that makes variable-resolution training work with standard batched tensors — do not attempt to pad/collate different-shaped images into one batch.
- Two separate optimizers (generator params vs discriminator params), typically Adam or AdamW, separate learning rates configurable
- Log per-step to **TensorBoard**: reconstruction loss, codebook loss, perceptual loss, adversarial loss (once active), discriminator loss (once active), codebook perplexity/usage, learning rate(s). Also log a per-bucket breakdown of these losses periodically — quality can differ meaningfully across buckets and this needs to be visible, not averaged away.
- Periodic image logging to **TensorBoard** (via `add_images`): save side-by-side input vs reconstruction grids every N steps, cycling through a few different buckets so all aspect ratios get visually sanity-checked, not just judged by loss numbers
- See **Checkpointing** section below for full requirements.

### Gradient Accumulation

- **`batch_size`** — how many images go into VRAM at once. This is limited by hardware and differs per bucket (e.g. the 1:1 bucket at 768×768 may only allow `batch_size=2`, while 16:9 at 768×448 may allow `batch_size=8`).
- **`accumulation_steps`** — how many batches to run before updating the model weights. Loss is accumulated over these steps instead of updating every single batch.
- A weight update happens every `batch_size × accumulation_steps` images. Since `batch_size` differs per bucket, `accumulation_steps` should be set per bucket too, so weight updates happen over roughly the same number of images regardless of which bucket is training.
- Log `batch_size` and `accumulation_steps` per bucket at the start of each run so it's clear from the log what each bucket is actually doing.

## Training Length — Epoch-Based

- **1 epoch = the model has seen every image in the training dataset once.** E.g. a 300,000-image dataset = 300,000 images seen per epoch, regardless of `batch_size`.
- `steps_per_epoch = dataset_size / (batch_size × accumulation_steps)` — computed at runtime from the actual dataset size, not hardcoded. Since buckets have different sizes and different `batch_size`/`accumulation_steps`, compute this per bucket and sum across buckets for the run total.
- The config knob is `target_epochs` (e.g. `30`), not a raw step count. Total steps for the run is derived from `target_epochs × steps_per_epoch`, recomputed whenever dataset size or batch settings change.
- Changing `batch_size` or `accumulation_steps` changes how many steps make up an epoch, but does **not** change `target_epochs` — the model still sees the same amount of data per epoch either way.
- One thing worth knowing (not something to act on now): when `batch_size × accumulation_steps` changes a lot, the learning rate sometimes needs adjusting too — bigger total batch sizes often tolerate a higher LR. Separate concern from epoch count, just flagging it in case LR instability shows up later.

## Checkpointing

Training on a single home GPU across multiple sessions (not a machine left running 24/7) — checkpointing is not optional polish, it's core to making this usable at all.

- **Save a full checkpoint every N steps** (config value, e.g. every 250–500 steps to start, tune based on how much progress is acceptable to lose). Each checkpoint must contain everything needed to resume training identically to an uninterrupted run:
  - Generator weights (encoder + quantizer/codebook + decoder)
  - Discriminator weights (once Phase B has started)
  - Generator optimizer state
  - Discriminator optimizer state (once Phase B has started)
  - Current step count, current epoch number, and position within the current accumulation cycle
  - Codebook EMA state (cluster sizes, EMA embedding averages) if EMA is enabled
  - Dead-code revival tracker state (usage counts since last revival check) so revival timing isn't reset by a resume
  - LR scheduler state (warmup position, etc.)
  - RNG state (torch, numpy, python random) for reproducibility across resume
  - Config/hyperparameters used for that run (codebook size, β, bucket table, `target_epochs`, etc.) — so a checkpoint is self-describing and resuming with a mismatched config fails loudly instead of silently corrupting training
- **Keep a rolling window of recent checkpoints** (e.g. last 3–5) plus **periodic permanent milestones** (e.g. every 5,000 or 10,000 steps, or at each epoch boundary, kept forever) — this avoids filling the disk with every single checkpoint while still protecting against a bad recent save (corruption, crash mid-write) losing all progress.
- **Always keep a `latest.pt` (or symlink/pointer) that points to the most recent complete, verified checkpoint** — the resume command should default to "resume from latest" with no manual path-hunting required.
- **Atomic writes**: write each checkpoint to a temp file first, then rename to the final filename only after the write completes successfully. A crash or Ctrl+C mid-save must never leave a corrupted checkpoint sitting at the path the resume logic will look for.
- **Resume must be the default, safe path**: running the training command again (e.g. `uv run python -m vqgan.train`) should auto-detect an existing `latest.pt` in the run directory and resume from it, printing what epoch/step it's resuming from — not silently start over from epoch 0. Starting fresh should require an explicit flag (e.g. `--fresh` / `--no-resume`), not be the default behavior.
- Log to console (and TensorBoard) whenever a checkpoint is saved and whenever a resume happens, including the epoch and step number, so it's obvious from the log alone what happened across a session boundary.
- Best-effort save on interrupt: catch `KeyboardInterrupt` (Ctrl+C) and attempt one final checkpoint save before exiting, so a manual stop doesn't lose progress since the last periodic save.
- Target training length: `target_epochs` in the 25–40 range as a starting expectation for COCO-minitrain-scale datasets (~25K images) — see Training Length section above. Make it configurable, not a hard stop, and recompute the derived total step count whenever dataset size changes (e.g. moving to COCO full).

## Progress Reporting

- Progress bar (tqdm or similar) must display **both epoch progress and step progress**, since epochs are the primary training-length unit: e.g. `Epoch 3/30 | step 4021/150000 (this run) | step 412000/4500000 (overall)`, or equivalent — epoch count must be visible, not buried behind a raw step counter the user has to mentally divide.
- Step counts shown (current/total) must be whole steps (post-accumulation), never fractional/partial-accumulation-cycle numbers.
- Step speed shown (s/step or it/s) must reflect time per full step (one accumulation cycle + `optimizer.step()`), using a rolling average (e.g. last 50 steps) rather than the most recent step alone — the first few steps include CUDA warmup / cuDNN autotune overhead and will skew a single-step estimate badly.
- ETA = `(total_steps_for_target_epochs - current_step) × rolling_avg_seconds_per_step`, where `total_steps_for_target_epochs` is the derived value from the Training Length section (recomputed if `target_epochs`, dataset size, or batch settings change). Display it in the progress bar alongside step speed.
- Log the current bucket, `batch_size`, `accumulation_steps`, dataset size, computed `steps_per_epoch`, and `target_epochs` at the start of each run (and whenever any of them change) so past runs are auditable from the log alone.

## VRAM Monitoring

- Log `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` to TensorBoard every step (or every N steps if per-step is too noisy) — this makes VRAM headroom visible over time in the dashboard rather than only being discoverable after a slowdown or OOM.
- If observed step time suddenly increases by a large factor (e.g. >5×) compared to the rolling average, log a warning to the console — this is the signature of VRAM spillover into shared/system memory (via PCIe), not a normal slowdown, and is worth surfacing immediately rather than silently absorbing into the rolling average.

## Evaluation

Implement an eval script separate from training that computes, on a held-out validation split:
- **rFID** (reconstruction FID) — primary metric, target < 2.0 as a quality bar based on current published VQGAN variants. Compute both an overall rFID and a per-bucket rFID, since a bucket with less training data (e.g. extreme ratios like 16:9) may lag behind the more common ones.
- **LPIPS** (average over val set, overall and per-bucket)
- **PSNR** (secondary, less important than the above two — do not over-optimize for it, high PSNR can coexist with blurry/perceptually-bad output)
- Codebook utilization (% of codebook entries actually used) — low utilization means codebook capacity is wasted and should trigger a warning in the eval report

## Dataset / Data Pipeline

- Build a generic image folder dataset loader (no dependency on a specific dataset's metadata format — VQGAN training doesn't need captions, just images)
- Recommended starting datasets (image-only, no captions needed for this stage):
  - **COCO minitrain** (~25K images) for pipeline smoke-testing first
  - **COCO 2017 full** (~118K images) for the actual training run
  - Note: COCO images average ~640×480 / 480×640 (a 4:3 / 3:4-ish ratio). Mapped into this bucket system they land near the 4:3 / 3:4 bucket (768×576 / 576×768), which is roughly a 1.2× upscale from their native resolution — much milder than the 1.6× upscale the earlier 1024px-max-side version required. Still worth a periodic visual check on reconstructions, but this is no longer the dominant quality ceiling it was before. Treat COCO as sufficient for validating that the pipeline and bucket system work correctly; consider a higher-resolution dataset (e.g. Unsplash Lite/Full) later if pushing back toward 1024px max side.
  - COCO's native aspect ratios cluster around 4:3/3:4 — it will not naturally populate the 1:1, 16:9, 9:16, or other buckets well. If bucket coverage matters for this training run (not just the common ratio), supplement with a source that has a wider natural spread of aspect ratios, or center-crop/pad a portion of COCO into underrepresented buckets deliberately.
- **Bucket assignment logic**: for each image, compute its aspect ratio and assign it to the closest bucket in the table above (by ratio, not by absolute size). Resize (preserving aspect ratio) then center-crop to the bucket's exact target resolution — do not naively stretch/squash to fit, since that distorts content.
- Preprocessing: resize+crop to assigned bucket resolution as above, standard augmentation (random crop within tolerance, horizontal flip) — no aggressive color augmentation, since the model needs to learn accurate color reproduction
- Data loading must support resuming mid-epoch (tie into checkpoint resume), and the bucket sampler's state (which images assigned to which bucket) should be deterministic/reproducible across resumes

## Package Structure

Organize as an installable/importable package, not a flat script dump:

```
vqgan/
  __init__.py
  models/
    encoder.py
    decoder.py
    quantizer.py
    discriminator.py
    vqgan.py          # wraps encoder+quantizer+decoder into one module
  losses/
    perceptual.py
    adversarial.py
    losses.py          # combines all loss terms per the schedule above
  data/
    dataset.py
    buckets.py         # bucket table, aspect-ratio assignment, bucketed sampler
  train.py             # main training entrypoint, argparse or config-file driven
  eval.py              # standalone evaluation script
  config.py            # dataclass or similar holding all hyperparameters, including the bucket table
configs/
  vqgan-multi.json      # the actual config file used to launch/resume training — see Config File Sync below
runs/                  # tensorboard log dir (gitignored)
pyproject.toml         # uv-managed, no pinned versions (see Constraints)
README.md
```

## Config File Sync

`configs/vqgan-multi.json` must always reflect the **actual** hyperparameters currently in use, not just whatever was set at launch — this file is the source of truth someone reads to know what a run is really doing.

- Any value that can be auto-tuned or overridden at runtime — per-bucket `batch_size`, `accumulation_steps`, `target_epochs`, LR, codebook size/β, `discriminator_start_step`, etc. — must be **written back to `configs/vqgan-multi.json`** whenever it changes, not left stale in the file while the running process silently uses a different value in memory.
- This applies in particular to per-bucket `batch_size` auto-tuning (e.g. reducing `batch_size` after hitting VRAM limits): once the run settles on working values, the file on disk should show those actual values, not just the initial guess.
- Same atomic-write requirement as checkpoints (see Checkpointing): write to a temp file, then rename — never leave `vqgan-multi.json` half-written if the process is interrupted mid-save.
- On resume, load config from `configs/vqgan-multi.json` by default (consistent with checkpoint resume being the default safe path) — CLI flags can override individual values for a given run, and any such override should also be written back so the file stays current.
- Log to console whenever `configs/vqgan-multi.json` is updated, so config changes are visible in the training log, not just silently applied.

## Constraints

- **Do not pin exact package versions** — in `pyproject.toml`, use unpinned or minimum-version (`>=`) specifiers only, so `uv sync` resolves current compatible versions. Commit `uv.lock` for reproducibility, but don't hand-pin versions in the dependency list itself.
- Framework: PyTorch (see Tooling section above).
- Code should run on a single GPU (no distributed training setup needed at this stage, but don't actively prevent it either — keep the training loop simple enough to extend later).
- Prioritize correctness and readability over premature optimization — this is an R&D first pass, not a production-hardened release.
- Architecture must be genuinely shape-agnostic (see Encoder/Decoder/Discriminator notes above) — this is the load-bearing requirement that makes the whole bucket system work with one set of model weights. Test this explicitly: run a forward+backward pass through at least two different buckets in the same test to confirm no shape assumptions leak in anywhere.

## Out of scope for this task

- The AR transformer that will later consume this VQGAN's tokens
- The text encoder (Qwen3.5) integration
- The **canvas / EMPTY-token system** for feeding the AR Model a fixed sequence length across buckets — that reconciliation happens at the AR Model layer, not here. This VQGAN's job ends at: given an image, assign it to the correct bucket and produce that bucket's token sequence; given a token sequence and its bucket, reconstruct the image. Making sequence length uniform across buckets for the AR Model is a separate, later task.