# Multi-GPU (DDP)

Code: `scripts/train_vqgan.py:269 :: main()`, `vqgan/training/distributed.py`

`setup_distributed()` returns `(False, 0, 0, 1)` unless torchrun launched the
process, in which case every count that multiplies by `world_size`
degenerates to what a single process computed before DDP existed.

## What is split and what is not

Only the **train** stream is split across processes. The val dataset is never
iterated as a `DataLoader` — `eval_subset.py` reads its shard list directly —
and only rank 0 evaluates anyway. See [eval-readout.md](eval-readout.md).

Rank 0 evaluates and writes; the others carry straight on to the next step and
block at its gradient all-reduce until rank 0 catches up. `evaluate()` gets
the unwrapped model, so nothing touches DDP's per-iteration bookkeeping
outside the training step itself.

## Wrapping order

The model is wrapped **after** any resume has been loaded, so DDP's
constructor broadcasts the resumed weights to every rank rather than a fresh
random init.

`apply_train_stage()` runs before the wrap too, since DDP reads
`requires_grad` when it builds its reducer. See
[resume-and-head.md](resume-and-head.md).

## broadcast_buffers=False

The one buffer set that has to agree across ranks — the quantizer's EMA
codebook — is synced explicitly in `VectorQuantizer.forward()`. DDP's own
per-forward broadcast would fight with that by reinstating rank 0's copy over
everyone's local update.

## The discriminator is a gradient path, not a party to the generator step

In the generator half of a step the generator needs `d(gan_loss)/d(recon)`,
never `d/d(disc weights)`, and `opt_d.zero_grad()` always threw the latter
away unused. Freezing the discriminator there skips computing them in the
first place — and under DDP it is **load-bearing** rather than an
optimization, because DDP's reducer allows each parameter to be marked ready
exactly once per iteration: letting `g_loss.backward()` reach discriminator
parameters that `d_loss.backward()` then reaches again trips "marked ready
twice".

For the same reason the adaptive-weight and GAN-loss forward passes use the
raw module rather than the DDP wrapper. DDP arms its reducer during forward
and then expects that module's gradients to arrive in the backward that
follows; they never will there, so going through the wrapper would leave the
reducer waiting on a reduction that never happens. The discriminator's own
step is the pass that legitimately syncs it.

## Keeping the ranks in step

Every counter that drives a trigger is identical on every rank, which is what
keeps the collectives those triggers lead to matched. `step_images` is
`batch * world_size` and needs no collective to compute, because
`drop_last=True` makes every rank's batch exactly `batch_size`. Details in
[clocks.md](clocks.md).

So every rank flips the EMA switch and the unfreeze at the same point, and the
collectives inside the quantizer's EMA branch stay matched.

## Learning rate

DDP averages gradients across ranks, so the batch a rate is being scaled for
is the **global** one and not this rank's share. See
[schedule-units.md](schedule-units.md).

## Writing checkpoints

`unwrap()` before `state_dict()`, so a DDP run writes the same plain keys a
single-GPU run does. DDP's own `state_dict()` would prefix everything with
`module.` and no other script in this project knows how to read that. See
[checkpoints.md](checkpoints.md).
