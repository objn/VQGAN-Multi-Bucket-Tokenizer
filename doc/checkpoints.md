# Checkpoint naming and contents

Code: `scripts/train_vqgan.py:867 :: save_checkpoint()`, `vqgan/checkpoints.py`

## Names

```
vqgan_step0028582.pt                      a model with no refinement head
vqgan_step0028582_refine_step0050000.pt   the same model, head at 50k steps
```

The first number is `global_step`, the model's total optimizer steps. The
second is `refine_global_step`, the head's own — see [clocks.md](clocks.md).
The refine half is absent when the model has no head, and that absence is
meaningful: the name says whether the file has a head and how much training
each half of it has had, without the file being opened.

Both counters only ever move forward across a model's whole life, so a name is
claimed once and the file under it is never rewritten.

## There is no vqgan_last.pt

A fixed name is a file that every run overwrites, and what it overwrites is
whatever the run before it spent hours producing — including, if that name was
ever used as a starting point, the run's own backbone. The menu's default for
attaching a head pointed at `vqgan_last.pt`, so the ordinary path destroyed
the thing it started from.

So "the newest checkpoint" is not a name to open, it is a question to ask the
directory. `vqgan/checkpoints.py:38 :: latest_step_checkpoint()` asks it.

It is kept out of `main.py` because the menu is not the only caller —
`evaluate`, `reconstruct`, `test_whole_image` and `visualize_model` all need
the same answer for their `--vqgan-checkpoint` default, and a CLI script
importing the interactive menu to get it would be backwards.

### Ordering

By the pair `(step, refine_step)`, which is a total order along any one
model's history because neither counter ever decreases: a `refine_only`
stretch leaves the first number alone and separates its files by the second,
and once the backbone is training again the first takes over.

Two unrelated lineages sharing a directory can still tie or cross. That is a
directory to split with `--checkpoint-dir`, not something a sort can fix.

Sorted on the parsed numbers rather than on the filename, since the zero
padding that makes those agree today would stop agreeing for a run that
outgrew seven digits — and not on mtime, which says when a file was last
copied about rather than what is in it.

`default_checkpoint()` falls back to a `vqgan_step*.pt` path that does not
exist rather than to `""`, so a run in an empty directory fails with a
missing-file error naming the place it looked instead of an empty-path one
that names nothing.

## The final save

The end of a run gets a checkpoint under its own step number, skipped when the
periodic save already wrote that exact step.

`last_saved_step` starts at whatever step the run arrived at — a `--resume`'s
`global_step`, or 0 for a fresh run. That is what makes a run doing no steps
write nothing: a `--resume` of a checkpoint already at `max_steps` breaks out
of the loop immediately, and without the guard would rewrite the very file it
was started from, under a fresh optimizer.

## Contents

| key | note |
| --- | --- |
| `global_step` | the filename's first number, and how pre-images checkpoints are read |
| `images_seen` | what the schedule actually runs on; two runs at different batch sizes agree on images, not steps |
| `vq_images_seen`, `refine_images_seen`, `refine_global_step` | the training clocks — see [clocks.md](clocks.md) |
| `ema_switched` | a plain Python attribute on the quantizer, not part of any `state_dict` |
| `model_config` | the architecture, read back rather than taken from `cfg` — see [resume-and-head.md](resume-and-head.md) |
| `vqgan`, `discriminator` | weights, via `unwrap()` |
| `opt_g` (with `param_names`), `opt_d` | see [optimizer-state.md](optimizer-state.md) |

`unwrap()` so a DDP run writes the same plain keys a single-GPU run does —
DDP's own `state_dict()` would prefix everything with `module.` and no other
script in this project knows how to read that.

`param_names` records what Adam's positional state keys refer to. Without it a
reader has to assume the parameter list has not changed shape since, which is
exactly the assumption that breaks across a stage or an EMA switch.

## Sizes

With the optimizer state carried across every stage
([optimizer-state.md](optimizer-state.md)), a checkpoint is ~2,081 MB
whatever stage wrote it. Measured breakdown of the 2,079 MB `joint` file at
dim 768 / depth 12 / codebook 16384x32:

```
vqgan weights        693.0 MB   (307 tensors)
discriminator          2.7 MB   (12 tensors)
opt_g Adam state    1377.4 MB   (302 params x exp_avg + exp_avg_sq)
opt_d Adam state       5.3 MB   (12 params)
```

Two thirds of the file is Adam's moments. Weights alone are 696 MB.

`--vqgan-checkpoint` starts the generator's optimizer fresh, so the first
checkpoint of a lineage begun that way is ~703 MB and grows to the full size
once the backbone has trained.
