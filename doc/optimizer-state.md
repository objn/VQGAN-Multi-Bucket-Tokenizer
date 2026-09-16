# The generator's optimizer, and carrying its state

Code: `scripts/train_vqgan.py:175 :: build_opt_g()`, `adopt_opt_g_state()`,
`flat_named_params()`

## betas

`betas=(0.5, 0.9)`, fixed. `(0.5, 0.9)` is the pairing GAN training is known
to be stable at, and compounding them per image the way the codebook EMA does
would leave essentially no momentum at all at large batches (`0.5 ** 16` is
1.5e-5). They are a momentum window measured in *steps*, so unlike everything
else in this project's schedule they do shift with batch size.

A consequence worth knowing when weighing whether any of this state is worth
preserving: `1/(1-0.5)` is 2 steps and `1/(1-0.9)` is 10. **Adam's memory here
is about ten steps.** That is why dropping it was survivable for as long as it
was, and why none of what follows is urgent on its own merits — it is worth
doing because it also buys a stable file size and a resume that works across
stages.

## Two groups, always the same parameters

The head gets its own param group, tagged `refine_group` so the training loop
can drive it from `RefineConfig`'s schedule; the rest of the generator is on
the other. Two independent schedules over two clocks — see
[clocks.md](clocks.md). The head's group is omitted only when the model has no
head at all, since Adam rejects an empty group.

**Parameters a stage has frozen go in anyway.** Adam skips any parameter whose
`.grad` is None, so a frozen one is held without being stepped and without so
much as an `exp_avg` allocated for it. Freezing is enforced by `requires_grad`,
not by optimizer membership, and the two say the same thing.

Verified directly: handing Adam four parameters with the first two frozen
gives `param_groups: [4]`, `state` for indices `[2, 3]` only, and the frozen
weights untouched.

## Why the frozen ones are handed over

Adam keys its state by each parameter's **position** in the optimizer's
groups. Filtering on `requires_grad` made that key space a function of
`train_stage`: index 0 was the encoder's position embedding in a `joint` run
and the head's first convolution in a `refine_only` one. Three things followed.

1. A resume across stages failed outright — the saved state described a
   different optimizer, and the error talked about param group sizes.
2. A `refine_only` run dropped the backbone's moments on the floor, and never
   got them back: a frozen backbone earns no new ones, so every refine
   checkpoint from then on carried none either.
3. Checkpoint size tracked the stage. A `refine_only` file was 703 MB against
   2,079 MB for a `joint` one, because `opt_g` is two thirds of the file and
   nearly all of it was absent.

With every parameter present the positions mean the same thing in every stage.
A `refine_only` run loads the backbone's moments, leaves them untouched for
the length of the run, and writes them back out. They are still there when the
backbone trains again, and every checkpoint is the same size.

The cost is that Adam walks the frozen parameters each step to skip them.

## No mid-run rebuilds

`build_opt_g` used to be called again at the EMA switch and at the unfreeze,
because both changed which parameters were trainable and therefore which were
in the optimizer. Both calls are gone.

- **EMA switch**: `set_use_ema()` clears the codebook's `requires_grad` so
  that EMA is the only thing writing it. Adam simply stops stepping a
  parameter it is still holding. Rebuilding used to be what took the codebook
  out of the optimizer, and it threw away every other parameter's moments to
  do it — on every run, whether or not a head was involved.
- **Unfreeze**: `opt_g` already holds the backbone, so clearing
  `requires_grad` is the whole of it. The backbone picks its moments up where
  the last run that trained it left off, and the head keeps its own across the
  switch instead of being reset at the exact moment it stops being the only
  thing training.

## Re-seating state on load

**Positional adoption is unsound.** A position is only meaningful next to the
list of parameters it was numbered against, and two things move it: adding a
head (the group list grows), and — before the filter was dropped — a stage or
an EMA switch excluding something from the middle.

The codebook is the case that bites. It leaves the optimizer once EMA takes
over, so reading a post-EMA-switch file positionally against today's full list
shifts every parameter after it by one. Measured on the real checkpoint:

```
index positions whose saved moment does NOT fit the parameter now there: 101
   (151, 'quantizer.codebook.weight', (1, 1024, 768), (16384, 32))
   (152, 'decoder.pos_embed',         (768, 32),       (1, 1024, 768))
   (153, 'decoder.from_code.weight',  (768,),          (768, 32))
```

101 parameters would have been handed each other's moments. Silently: the
shapes are wrong, but nothing checks them, and Adam would carry on with
moments that describe a different tensor.

So state is re-seated **by name**:

- Checkpoints written from here on store `param_names` — the flattened name
  list in group order, so position *i* in the state is `param_names[i]`.
  Names are matched to today's positions. Anything the file has that this
  model does not is dropped; anything new, such as a freshly attached head,
  starts without state.
- Checkpoints written before that carry no names, and their positions have to
  be reconstructed. Only the exclusions the old `build_opt_g` could actually
  produce are tried — it filtered on `requires_grad`, so the candidates are
  "everything" and "everything but the codebook" — and each is accepted only
  if every saved moment's shape matches the parameter it would land on. A file
  that fits neither is refused rather than guessed at.

The group *description* that gets loaded is always this run's, not the file's:
it has the head's group, and the rates on it are overwritten from the schedule
every step anyway. A file's saved `param_groups` carry the rate it left off
at, which for a run starting again at image 0 would be a decayed one.

## opt_d

Untouched by all of this. The head is on the generator side,
`PatchDiscriminator`'s shape is identical either way, and the discriminator
trains every step whatever `train_stage` says — so resetting it would only
throw away a trained adversary for no reason.

## --vqgan-checkpoint

Still starts the generator's optimizer fresh, by definition: it takes weights
alone and begins a new run around them. See
[resume-and-head.md](resume-and-head.md).
