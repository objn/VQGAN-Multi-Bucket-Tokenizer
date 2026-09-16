# The in-training readout

Code: `scripts/train_vqgan.py:813 :: evaluate()`, `main()`;
`vqgan/display.py:50 :: TrainingDisplay`

## The evaluation subset

A fixed, size-balanced subset of the validation split, computed once at
startup (header reads only) and decoded once, so every `evaluate()` call for
the rest of the run reuses the same crops instead of re-scanning the split.
See `VQGANTrainConfig.eval_images`, `eval_size_groups`, `eval_prep_file` and
`vqgan/data/eval_subset.py`.

Only rank 0 ever calls `evaluate()`, so only rank 0 builds any of it — the
others would be decoding thousands of crops to throw them away.

The val stream uses the same crop size and the same grid as training, minus
the per-pass jitter that rides along with `shuffle`, so the *n*th validation
crop is the same crop at every evaluation point. Comparable to itself across
the run, in other words, not to anything else.

`preview_images` is one representative image per size group — the first image
the round-robin picked from each group, reusing its already-decoded crops (all
of them, not just one) rather than a separate pick. It can run well past
`batch_size` crops, since a large-group image may hold dozens, so it gets the
same chunked treatment as the main readout.

Run under the same bf16 autocast as training. In fp32 this was 2.6x more
expensive for no useful precision: 2,048 crops took 79.6s against 30.3s, and
the val L1 agreed to four decimals (0.1469 either way on the 40k checkpoint).
Codebook usage does move a little — 99.1% vs 97.4% on that same pass, since
rounding flips which code a few tokens land on — so read it as a health
indicator, not an exact figure.

## What the readout can and cannot see

Worth knowing before tuning `--refine-eval-every-steps` down on a short refine
run: this reports val L1 and codebook statistics, and **the head moves
neither much**. L1 barely registers a seam being removed, and under
`refine_only` the codebook is frozen outright, so its numbers sit still by
construction.

The `recon_img*.png` previews written alongside are where the 8x8 patch grid
either is or is not, and `scripts/test_whole_image.py` is the before/after.

## Codebook metrics

Usage answers "how many codes are ever touched"; perplexity answers "how many
are actually carrying the representation". A codebook can score ~100% on the
first while a handful of codes take nearly every lookup. See
`VectorQuantizer.codebook_perplexity()`.

`usage_count` is reset at the start of every `evaluate()`, so its value is
"usage during this eval pass".

## The display rows

Four rows that redraw in place rather than one growing line. The training loop
reports ten-odd numbers; as a single tqdm postfix they ran past the width of a
tmux pane, and a bar that cannot fit its line stops overwriting and starts
scrolling, which buries the run's actual log (eval results, the EMA switch,
checkpoints) in redrawn duplicates.

```
train  27% ---------------- 902,852/3,400,000   72.5 img/s eta 9:33:41
loss   recon 0.0321  laplace -1.9477  lpips 0.3510  vq 0.0007  d_w 9.31
rate   vq 7.07e-06 @ 1,400,004 img   head 1.21e-06 @ 400 img
code   14,203/16,384 used (86.7%)  perplexity 9,842/16,384 (60.1%)
```

Each is updated on its own cadence and re-measured against the terminal width
on every refresh, so a resized pane re-wraps instead of smearing. Permanent
log lines still go through `console.print`: rich's `Live` moves the block down
and prints above it, so the history stays readable and scrollable.

Off a TTY (`> log.txt`) there is nothing to redraw in place, so the live block
is skipped entirely and each update prints one plain line — a log file of one
row per interval rather than thousands of escape sequences.

### Why the rates get their own row

They are not measured against the same thing. The bar counts the run; each
rate is a position on its own schedule, read against its own image count
([clocks.md](clocks.md)). Under `refine_only` the backbone's clock is standing
still while the head's advances, so a rate printed without the count it was
read at is a number no one can check. On the loss row there was nowhere to put
those counts, and the bar's single position would have been read as belonging
to both.

### Only live rates are shown

`set_losses(logs, rates)` takes one entry per half of the model that is
actually training, each carrying the image count its rate was read against.
Halves without gradients get no entry.

| stage | shown |
| --- | --- |
| `refine_only` | `head` only |
| `vq_only`, no head | `vq` only |
| `joint` | both |

A learning rate belonging to frozen parameters is worse than no number at all,
because it is exactly the number a reader watches to decide whether the
schedule is working. Under `refine_only` the generator's rate goes on being
computed and decaying beside a head that is the only thing training — and a
single `lr` column next to the losses was that number. It is what hid a refine
run whose head never annealed.

TensorBoard follows the same rule: `train/lr` and `train/refine_lr` are
written only while the half they belong to is training. Both clocks are logged
too, so a curve read later can be placed against the age of the half of the
model that produced it rather than against the run that contained it.

The x-axis is images, not steps, so curves from runs at different batch sizes
lie on top of each other instead of being stretched apart by a factor of
batch.

### Which losses are on screen

`d_weight` is shown to 8 places: the adaptive weight is a gradient-norm ratio
that routinely sits in the 1e-3..1e-2 range, where two places round it to a
flat `0.00` and hide the thing being watched.

`g_loss`, `d_loss` and the two grad norms go to TensorBoard in full but not to
the screen — they are what you read when something has already gone wrong,
not every tick, and each one on screen costs width the other fields use
better.

The codebook row deliberately holds its last values between evals rather than
blanking out, since usage and perplexity are only measured there.
