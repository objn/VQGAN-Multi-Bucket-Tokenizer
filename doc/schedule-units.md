# The schedule is counted in images

Code: `scripts/train_vqgan.py:35 :: cosine_lr()`, `crossed()`,
`vqgan/config.py:34 :: VQGANTrainConfig`

Every part of the schedule — `max_steps`, the warmups, the eval and checkpoint
cadence, the LR decay — is counted in **images**, not in optimizer steps at
whatever `batch_size` the run uses.

`batch_size` is therefore a pure throughput knob: warmups, evaluation,
checkpoints and the decay all land at the same point in the data whatever it
is set to, and a bigger batch just gets there faster.

Epochs are too coarse a unit to schedule on — one pass over ImageNet-1k is
~3.8M crops at the current tiling.

Steps are still counted, but only to name checkpoints
([checkpoints.md](checkpoints.md)) and to average the running loss over the
batches that produced it.

## crossed()

```
crossed(previous, current, every) == (current // every > previous // every)
```

A batch rarely lands exactly on a multiple, so `count % every == 0` would skip
whole triggers at larger batch sizes — at batch 128, only one image count in
128 is even a candidate. Comparing floor divisions fires once per interval
crossed no matter how the images were grouped. It is the same test the
quantizer uses for dead-code revival.

## cosine_lr()

Linear warmup from 0 to `base_lr` over the first `warmup_images`, then cosine
decay to `min_lr` over the images remaining until `end_images`. Never below
`min_lr`, and flat at `min_lr` for anything past `end_images` — which is where
the decay ends, not necessarily where training ends.

```
progress = (images_seen - warmup) / (end_images - warmup - 1)
lr       = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
```

Progress is measured in images rather than steps so the curve a run follows is
the same curve at any batch size. `warmup_images=0` skips straight to the
cosine at `base_lr` on image 0.

**The cosine is flattest at its start** — the derivative at `progress = 0` is
zero. That is why an endpoint far past the end of a run is a constant learning
rate wearing a schedule's clothes: against a 3.4M endpoint, a 200k-image run
walks 5.6% of the curve and its rate falls 0.8%.

| `images_seen` | `progress` | lr (base 1e-4, end 3.4M) | lr (end 200k) |
| --- | --- | --- | --- |
| 10,000 | 0.000 | 1.000e-4 | 1.000e-4 |
| 50,000 | 0.012 | 9.997e-5 | 8.95e-5 |
| 100,000 | 0.027 | 9.983e-5 | 5.46e-5 |
| 200,000 | 0.056 | 9.924e-5 | 1.00e-6 |

Which clock each curve is read against is in [clocks.md](clocks.md).

## Batch-size scaling of the rate itself

Separate from the schedule, and often confused with it.
`VQGANTrainConfig.scaled_lr()` converts the rate you *quote* into the rate the
run *uses*, given its batch. A rate quoted anywhere therefore means the rate at
`reference_batch_size`, and the same number stays comparable across runs at
different batches. At `reference_batch_size` there is no scaling at all.

This is why the menu prompt reads `at batch 8, scaled from there` even when you
just answered 4 for batch size: the 8 is `reference_batch_size`.

Three rules, with `ratio = batch_size * world_size / reference_batch_size`:

| `lr_scaling` | multiplier |
| --- | --- |
| `"sqrt"` (default) | `ratio` below the reference batch, `sqrt(ratio)` at or above it |
| `"linear"` | `ratio` |
| `"none"` | 1 |

## Why "sqrt" is piecewise

What matters is not the size of a step but how far the weights travel per
image of data. Every loss in this project uses **mean** reduction, so the
gradient's expected magnitude does not depend on the batch — and Adam
normalizes by a running gradient magnitude on top of that, so a step moves
roughly `lr` whatever the batch is. What the batch changes is how many steps
a given number of images buys:

```
travel per image  ∝  lr / batch
```

Halve the batch and you take twice as many steps of the same size over the
same data. That is what the scaling exists to correct, and it is the reason
mean reduction makes the correction *necessary* rather than redundant.

A plain `sqrt(ratio)` only corrects half of it, in log terms. Above the
reference batch that errs safe — the rule is increasingly conservative, which
is the intended trade against a `linear` rule that pushes large batches into
the range where a ViT plus an adversarial term goes unstable (`linear` at
batch 128 asks for 1.6e-3). Below the reference batch the same shortfall errs
the *other* way, and training at batch 4 or less was visibly unstable:

| batch | `sqrt(ratio)` | travel/image | piecewise | travel/image |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.354 | **2.83x** | 0.125 | 1.00x |
| 2 | 0.500 | **2.00x** | 0.250 | 1.00x |
| 4 | 0.707 | **1.41x** | 0.500 | 1.00x |
| 8 | 1.000 | 1.00x | 1.000 | 1.00x |
| 16 | 1.414 | 0.71x | 1.414 | 0.71x |
| 32 | 2.000 | 0.50x | 2.000 | 0.50x |
| 128 | 4.000 | 0.25x | 4.000 | 0.25x |

So each rule is used where it is right: `linear` holds travel per image
constant and is the SGD result (Goyal et al.), sound while a batch is small
enough that step count rather than gradient noise is the limit; `sqrt` takes
over once it is not. **At or above the reference batch the multiplier is
unchanged**, so nothing about a run at batch 8 or more differs from before
this rule existed.

`betas=(0.5, 0.9)` sharpens all of this — those are windows of about 2 and 10
steps, so Adam smooths very little here and per-step noise passes almost
straight through. See [optimizer-state.md](optimizer-state.md).

One thing the rule does not cover: crops are tiles of a photo, so a small
batch may hold several tiles of the same image and be worth fewer independent
samples than its size suggests. `shuffle_buffer` is set to at least
`8 * batch_size` to fight exactly that. `grad_clip_norm` is not covered
either — a noisier small-batch gradient hits the ceiling more often, which
shortens steps in a way no learning-rate rule accounts for.

## The rest of the arithmetic

Under DDP the batch a rate is scaled for is the **global** one, since DDP
averages gradients across ranks — so `world_size` is passed in, and an N-GPU
run at a given global batch picks the same rate a 1-GPU run at that batch
would. 4 GPUs at batch 2 and 1 GPU at batch 8 both resolve to the quoted rate.

The head has its own rule (`RefineConfig.lr_scaling`) put through the same
arithmetic. Sharing the arithmetic is not sharing the choice.

`scaled_lr` lives on the config rather than in the training loop so the number
a run uses is derivable from its config alone — a checkpoint's config is the
only record of what it was trained at.

`--refine-lr-scaling` is validated in `parse_args()` rather than left to
`scaled_lr()`, which would not raise until the run was already several seconds
into building a model.
