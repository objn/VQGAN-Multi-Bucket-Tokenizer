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
run *uses*, given its batch:

```
lr_used = lr_quoted * sqrt(batch_size * world_size / reference_batch_size)
```

with `"sqrt"`, `"linear"` and `"none"` as the three rules. So a rate quoted
anywhere means the rate at `reference_batch_size`, and the same number stays
comparable across runs at different batches. At `reference_batch_size` there
is no scaling at all.

This is why the menu prompt reads `at batch 8, scaled from there` even when
you just answered 4 for batch size: the 8 is `reference_batch_size`, and
`1e-5` typed at batch 4 becomes `7.07e-6`.

Under DDP the noise this corrects for is set by the **global** batch, since
DDP averages gradients across ranks — so `world_size` is passed in and an
N-GPU run at a given global batch picks the same rate a 1-GPU run at that
batch would.

The head has its own rule (`RefineConfig.lr_scaling`) put through the same
arithmetic. Sharing the arithmetic is not sharing the choice.

`scaled_lr` lives on the config rather than in the training loop so the number
a run uses is derivable from its config alone — a checkpoint's config is the
only record of what it was trained at.

`--refine-lr-scaling` is validated in `parse_args()` rather than left to
`scaled_lr()`, which would not raise until the run was already several seconds
into building a model.
