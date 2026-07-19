# VQGAN Implementation Spec — Multi-Resolution Bucket Tokenizer (max side 1024)

## Context

This VQGAN is the image tokenizer for a larger AR text-to-image pipeline (Qwen3.5 text backbone + custom AR transformer, not part of this task). This task covers **only the VQGAN tokenizer**: encoder, quantizer, decoder, discriminator, losses, and training loop. Build it as a clean, importable Python package, not a notebook.

Target hardware: single NVIDIA RTX 3080 Ti, 12GB VRAM, 64GB system RAM (DDR4). Design every default (batch size, precision, checkpointing) around fitting this GPU.

## Architecture Requirements

### Resolution / bucket system

This VQGAN must handle **multiple aspect ratios**, not a single fixed square size. The constraint is: **longest side = 1024px**, with the other side determined by aspect ratio, both sides rounded to the nearest multiple of 32 (the downsampling factor).

Downsampling factor: **32×** (5 downsampling stages of ÷2 each: 2^5 = 32). This is fixed regardless of bucket.

Bucket table (compute more at runtime as needed, but these are the reference set to support first):

| Ratio | Resolution (px) | Grid   | Tokens |
|-------|------------------|--------|--------|
| 1:1   | 1024×1024        | 32×32  | 1024   |
| 4:5   | 832×1024         | 26×32  | 832    |
| 5:4   | 1024×832         | 32×26  | 832    |
| 4:3   | 1024×768         | 32×24  | 768    |
| 3:4   | 768×1024         | 24×32  | 768    |
| 3:2   | 1024×672         | 32×21  | 672    |
| 2:3   | 672×1024         | 21×32  | 672    |
| 16:9  | 1024×576         | 32×18  | 576    |
| 9:16  | 576×1024         | 18×32  | 576    |

Because token count differs per bucket, **each training batch must contain images from a single bucket only** (see Data Pipeline below) — this is standard "aspect ratio bucketing" as used in SDXL and similar systems.

- Encoder and decoder must be **fully convolutional with no fixed-size assumptions** (no `nn.Linear` or `nn.Flatten` anywhere that hardcodes a spatial dimension, no hardcoded H/W in `forward()`). This is what lets the same encoder/decoder weights handle every bucket in the table without any architecture changes.
- Attention blocks (encoder bottleneck, decoder bottleneck) must work with variable sequence length — standard self-attention already does this, just don't hardcode a sequence length anywhere (e.g. no fixed positional embedding table sized for one specific grid — use a form of positional encoding that generalizes across grid shapes, such as 2D RoPE or a resolution-agnostic scheme).
- Encoder and decoder must be symmetric (mirrored architecture) at every bucket.

### Encoder
- CNN-based (ResNet-style residual blocks + downsampling convs), following the VQGAN (Esser et al. 2021) / Taming Transformers design
- 5 downsampling stages, each halving spatial resolution (e.g. for the 1:1 bucket: 1024 → 512 → 256 → 128 → 64 → 32)
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

- Mixed precision (bf16) — required to fit 1024px-class training on 12GB VRAM
- Gradient checkpointing on encoder/decoder — make it a toggleable flag
- Gradient accumulation support (target effective batch size larger than what fits in VRAM at once; expect physical batch size of 2–4 images at the 1:1 (1024×1024) bucket, more at smaller buckets like 16:9)
- **Bucketed batch sampler**: a custom `Sampler` (or dataset wrapper) that groups images into their nearest bucket (by aspect ratio) and only yields batches where every image belongs to the same bucket. This is the mechanism that makes variable-resolution training work with standard batched tensors — do not attempt to pad/collate different-shaped images into one batch.
- Two separate optimizers (generator params vs discriminator params), typically Adam or AdamW, separate learning rates configurable
- Log per-step to **TensorBoard**: reconstruction loss, codebook loss, perceptual loss, adversarial loss (once active), discriminator loss (once active), codebook perplexity/usage, learning rate(s). Also log a per-bucket breakdown of these losses periodically — quality can differ meaningfully across buckets and this needs to be visible, not averaged away.
- Periodic image logging to **TensorBoard** (via `add_images`): save side-by-side input vs reconstruction grids every N steps, cycling through a few different buckets so all aspect ratios get visually sanity-checked, not just judged by loss numbers
- Checkpoint saving/resuming (model + optimizer + step count), since this will run over multiple sessions
- Target total training length: 200,000–500,000 steps as a starting expectation; make this configurable, not a hard stop

## Progress Reporting

- Progress bar (tqdm or similar) must display **effective steps** (post-gradient-accumulation), never fractional/physical sub-steps. Format: `current_effective_step / total_effective_steps`, both integers.
- Step speed shown (s/step or it/s) must reflect time per **effective step** (i.e. time for one full accumulation cycle + `optimizer.step()`), using a rolling average (e.g. last 50 steps) rather than the most recent step alone — the first few steps include CUDA warmup / cuDNN autotune overhead and will skew a single-step estimate badly.
- ETA = `(total_effective_steps - current_effective_step) × rolling_avg_seconds_per_effective_step`. Display it in the progress bar alongside step speed.
- **`total_effective_steps` must be a fixed target independent of batch size.** If `accumulation_steps` is changed later (e.g. to fit VRAM after lowering physical batch size), `total_effective_steps` must NOT change — it represents a fixed amount of data the model should see, not a fixed number of physical batches. Physical batch size and accumulation_steps are free to vary as long as `physical_batch_size × accumulation_steps` (the effective batch size) stays constant.
- Log the current bucket, physical batch size, accumulation_steps, and resulting effective batch size at the start of each run (and whenever they change) so past runs are auditable from the log alone.

## VRAM Monitoring

- Log `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` to TensorBoard every step (or every N steps if per-step is too noisy) — this makes VRAM headroom visible over time in the dashboard rather than only being discoverable after a slowdown or OOM.
- If observed step time suddenly increases by a large factor (e.g. >5×) compared to the rolling average, log a warning to the console — this is the signature of VRAM spillover into shared/system memory (via PCIe), not a normal slowdown, and is worth surfacing immediately rather than silently absorbing into the rolling average.



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
  - Note: COCO images average ~640×480 / 480×640 (a 4:3 / 3:4-ish ratio). Mapped into this bucket system they land near the 4:3 / 3:4 bucket (1024×768 / 768×1024), which is roughly a 1.6× upscale from their native resolution. This is a real quality ceiling worth being aware of — reconstructions will be limited by this upscaling no matter how well the VQGAN trains. Treat COCO as sufficient for validating that the pipeline and bucket system work correctly; consider a higher-resolution dataset (e.g. Unsplash Lite/Full) before treating final reconstruction quality numbers as representative.
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
runs/                  # tensorboard log dir (gitignored)
pyproject.toml         # uv-managed, no pinned versions (see Constraints)
README.md
```

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
