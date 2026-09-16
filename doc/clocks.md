# The three clocks

Code: `scripts/train_vqgan.py:269 :: main()`

"How far along is this run" and "how much training has this half of the model
actually had" stop being the same question the moment a stage freezes
something. So there are three counters, not one.

| counter | counts | stands still |
| --- | --- | --- |
| `images_seen` | every image pulled from the loader, whatever it was used for | never |
| `vq_images_seen` | images that moved the backbone | under `refine_only` |
| `refine_images_seen` | images that moved the head, from the moment it was created | under `vq_only`; is 0 on a model with no head |

**Each schedule reads the clock of the thing it schedules.** A warmup then
measures the age of the parameters it is ramping rather than the age of the run
that happens to contain them.

- `images_seen` drives the eval, checkpoint and log cadence, and the `img/s`
  rate. It counts work done, which is what those are about.
- `vq_images_seen` drives the VQ learning-rate curve and the EMA switch.
- `refine_images_seen` drives the head's learning-rate curve, its
  discriminator warmup, and `unfreeze_steps`.

`refine_global_step` rides along beside `refine_images_seen`, counting
optimizer steps rather than images. It is not a schedule input — it is the
second half of the checkpoint filename, the counterpart of `global_step`. See
[checkpoints.md](checkpoints.md).

## Why the head needed its own

Read against `images_seen`, the head's schedule was read against the
backbone's age. A head created on a model 1.4M images in arrived past every
warmup it had, so `--refine-lr-warmup-steps` did nothing at all on the path
that creates one, whatever it was set to. Zero-init made that less alarming
than it sounds — the head begins as an exact identity — but the ramp was
simply absent.

`--refine-end-steps-lr` had the same problem in reverse: left at the VQ
default of 3.4M, a 200k-image refine run walked the first 6% of the cosine and
ended at ~99% of its base rate. Adversarial training has no minimum to settle
into, so the thing that stops it oscillating is the rate getting small. An
endpoint far past the end of the run means that never happens, and the grain
the head learns to add never anneals away.

## Budgets

Each budget stops against its own clock, and whichever is spent first ends the
run:

- the backbone at `VQGANTrainConfig.max_steps` on `vq_images_seen`
- the head at `RefineConfig.max_steps` on `refine_images_seen`

The backbone's is an **absolute position** it has been walking towards since
image 0. The head's is a **length**, counted from the moment it was created.
That difference is why they cannot share a field: asking for 200,000 images of
refinement on a 1.4M-image model would mean typing 1,600,004, a number that is
mostly the backbone's history — and typing the 200,000 you meant would end the
run before its first step, because the clock is already past it.

Under `refine_only` only the head's budget can be reached, since
`vq_images_seen` is standing still. That is what makes `RefineConfig.max_steps`
the length of a refine run.

Neither budget is measured on `images_seen`, which counts work done rather
than progress along either schedule.

## The progress bar

The bar tracks the budget that will actually end the run, on that budget's own
clock. A `refine_only` run drawn against the VQ pair would open at
`1,400,004/3,400,000 — 41%` and finish at 43%, describing a backbone that
never moves while the thing being trained went from 0 to done.

It advances by `step_images` either way: every step feeds the whole batch to
whichever half is training, so the clock it is drawn against advances by the
same amount `images_seen` does.

## Counting under DDP

`step_images` is `batch * world_size` — the images *the run* consumed this
step, every rank's batch and not just this one's. No collective is needed:
`drop_last=True` makes every rank's batch exactly `batch_size`, so the total
is arithmetic. It also keeps the counters identical on every rank, which is
what keeps the triggers that depend on them — and therefore the collectives
those lead to — matched across ranks. See [distributed.md](distributed.md).

`train_stage` is re-read every step when advancing the clocks, because
`unfreeze_steps` can change it mid-run.

## Stage transitions read the right clock

The **EMA switch** fires on `vq_images_seen`. The warmup exists to let the
encoder stabilize before the codebook starts following it, and a `refine_only`
stretch stabilizes nothing — the encoder is frozen and the quantizer is in
eval mode throughout. Counting those images would let the switch fire in the
middle of one.

The **unfreeze** (`refine_only` → `joint`) fires on `refine_images_seen`:
settle the head against a frozen backbone first, then let the two adapt to
each other. Every rank crosses it at the same value, so the freezing stays
consistent across ranks.

Neither rebuilds the optimizer any more — see
[optimizer-state.md](optimizer-state.md).

## Persistence

All three clocks go into every checkpoint. They cannot be recovered from
`images_seen` afterwards: how much of a run went to each half depends on the
stages it passed through, and nothing else records that.

Checkpoints written before the clocks were split carry only `images_seen`.
For the backbone that is the right answer — nothing could freeze it then, so
every image it saw trained it. The head's clock defaults to 0 unless the file
already had a head, and a file that had one without recording its age was
written by a run whose whole image count went to it.
