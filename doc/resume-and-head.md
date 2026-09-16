# Resuming, and adding the refinement head

Code: `scripts/train_vqgan.py:269 :: main()`, `merge_refine_config()`,
`apply_train_stage()`, `set_train_mode()`

## Two flags, one file

Both name a checkpoint to start from, and they differ in how much of it they
take. `parse_args()` rejects being given both.

| | takes | image count |
| --- | --- | --- |
| `--resume` | weights, optimizers, image count, EMA state | continues |
| `--vqgan-checkpoint` | the weights alone | restarts at 0 |

`--vqgan-checkpoint` starts a new run around borrowed weights. It exists
because the generator's optimizer state saved next to those weights may not
describe the parameter set this run trains — see
[optimizer-state.md](optimizer-state.md).

Either way the **shape of the network is a fixed property of the file** and is
read back from it rather than from `cfg`, whose defaults drift between runs.
Otherwise `load_state_dict` fails with a shape mismatch the moment the two
disagree. `merge_refine_config()` is the one seam where this run gets a say,
and the only thing it may say is: attach a head the checkpoint does not have.

## Adding a head is a resume, not a fork

A checkpoint without a head gets one; a checkpoint with one keeps training it;
and which of those is happening is read off the file rather than asked. Either
way the image count, the lr schedule and the step number continue — one model,
one lineage.

It used to be forced onto `--vqgan-checkpoint`, which meant the model forgot
it had ever trained: the counters went back to 0, the lr schedule restarted,
and checkpoint numbering began again from a point that could collide with
files already on disk. The only reason was that `opt_g` could not be loaded
across the change in parameter set. That is handled now.

The head is **zero-initialized**, so the model reconstructs exactly as it did
a moment before until the head is actually trained. Nothing is given up by
attaching one early.

`head_present` is settled the moment `model_config` is. It is a property of
the model, not of `train_stage`: a `vq_only` run on a model with a head still
has one.

### The one thing allowed to be missing

A head this run is attaching for the first time is the only key that may be
absent from the checkpoint. `strict=False` alone would also wave through a
genuinely mismatched checkpoint, so the keys it lets slide are checked by
hand: anything missing outside the head, or anything unexpected at all, is
still an error.

## Train stages

`--refine-train-stage` decides which half gets gradients.

| stage | backbone | head |
| --- | --- | --- |
| `joint` | trains | trains |
| `refine_only` | frozen | trains |
| `vq_only` | trains | frozen |

`joint` is spelled out rather than skipped as a no-op, because it is not one
after a stage change.

Applied **before** DDP wraps the model, since DDP reads `requires_grad` when
it builds its reducer, and before `opt_g`, which is built from the result.

`refine_only` also holds the quantizer in **eval mode**, not just clear of
gradients. The codebook's EMA update and its dead-code revival are gated on
`self.training`, not on `requires_grad`, so clearing gradients alone would
leave the codebook drifting underneath a head that is being trained against a
backbone which is supposed to be standing still.

One readout consequence: with EMA updates off, the quantizer falls back to
reporting its gradient-mode `vq_loss` (codebook + commitment rather than
commitment alone), so the logged `vq` number jumps when a run crosses between
`refine_only` and `joint`. Neither term reaches a frozen encoder or a codebook
that is out of the optimizer — it is a readout changing units, not a loss
changing behaviour.

`ema_on` is what stops `apply_train_stage` from undoing the EMA switch: once
the quantizer updates its codebook from EMA buffers that weight is out of the
optimizer for good, and handing it a gradient back would leave two mechanisms
writing the same tensor.

## EMA state on resume

`ema_switched` is a plain Python attribute, not part of any `state_dict`, so
it is restored directly — **not** via `quantizer.set_use_ema()`, which would
wipe the just-loaded EMA buffers thinking it was switching mode for the first
time.

On the `--vqgan-checkpoint` path it is the one thing that does *not* restart:
`global_step` and `images_seen` stay at 0 because the schedule should start
from the beginning, but `ema_switched` is a property of the codebook that came
in the file, and putting a converged codebook back through the gradient-update
warmup would only unsettle it.

## Which config the run is trained by

A model with a head is trained by `RefineConfig`'s copies of the loss,
adversarial and eval settings; a model without one by `VQGANTrainConfig`'s.
The field names match exactly, so the choice is one binding (`knobs`) rather
than a fallback at each use. **Nothing is merged: picking a side picks all of
it.**

That is why a refine run prints its loss weights on startup — they are the
`--refine-*` values, and a `--disc-weight` or `--lpips-weight` passed to such
a run was read into a config it is not training from.

## Usage

```bash
# plain training, and the same schedule at a bigger batch
python scripts/train_vqgan.py --max-steps 1600000
python scripts/train_vqgan.py --batch-size 128

# continue a run
python scripts/train_vqgan.py --resume checkpoints/vqgan_step0028582.pt

# add a refinement head and train it alone, keeping the image count and the
# schedule the checkpoint arrived with
python scripts/train_vqgan.py --resume checkpoints/vqgan_step0028582.pt \
    --refine-enabled true --refine-train-stage refine_only

# the same head, but on a new run whose schedule restarts at image 0
python scripts/train_vqgan.py --vqgan-checkpoint checkpoints/vqgan_step0028582.pt \
    --refine-enabled true --refine-train-stage refine_only
```
