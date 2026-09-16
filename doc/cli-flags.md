# How the training flags are generated

Code: `scripts/train_vqgan.py:57 :: parse_args()`

Every flag is derived from a dataclass field, so adding a config knob adds its
flag. There is no second list to keep in sync.

```
VQGANTrainConfig.batch_size   ->  --batch-size
VQGANTrainConfig.max_steps    ->  --max-steps
```

`bool` fields get a parser that treats anything but the literal `false` as
true, so `--refine-enabled true` and `--refine-enabled 1` both work and
`--refine-enabled false` is the only way to turn one off. Everything else is
parsed as `type(default)`.

## Nested blocks

`RefineConfig` is a nested dataclass on `VQGANTrainConfig`, and the loop that
walks plain fields cannot derive a flag from it. Nested blocks get their own
pass under a `--<block>-<field>` prefix:

```
RefineConfig.lr           ->  --refine-lr
RefineConfig.max_steps    ->  --refine-max-steps
RefineConfig.disc_weight  ->  --refine-disc-weight
```

`--refine-lr` then stays visibly separate from `--lr`, which is the point of
keeping the block separate to begin with. See
[resume-and-head.md](resume-and-head.md) for why a run picks one side's
settings wholesale rather than merging them.

## The two hand-written flags

`--reset-discriminator` and `--vqgan-checkpoint` are not config fields —
neither describes the model or its schedule, so neither belongs in a
checkpoint. They describe what this invocation should do with a file. See
[resume-and-head.md](resume-and-head.md).

## REFINE_PREFIX

`decoder.refine.` — the refinement head's parameters, by name, in every
`state_dict` and `named_parameters()` the script walks. One constant, so the
freezing, the optimizer split and the state-dict check cannot drift apart.

## Validation done here

`--refine-train-stage` and `--refine-lr-scaling` are checked in `parse_args()`
rather than where they are used, which would not raise until the run was
already several seconds into building a model. Both go through
`parser.error()`, so a bad value prints usage and exits rather than raising
into the interactive menu — which would take the whole menu down with it. The
menu validates the stage itself for the same reason, before calling in.
