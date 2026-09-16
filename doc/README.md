# doc/

Why the code is the way it is. One file per topic; the `.py` files keep a
one-line pointer at each place a topic applies.

[doc-refs.md](doc-refs.md) explains the reference scheme and the checker that
keeps the line numbers on both sides accurate. Run it before committing:

```
python scripts/doc_refs.py --fix
```

## Topics

| file | about |
| --- | --- |
| [clocks.md](clocks.md) | the three image counters, which schedule reads which, and the budgets that end a run |
| [optimizer-state.md](optimizer-state.md) | why `opt_g` holds frozen parameters, and how its state is carried across stages |
| [checkpoints.md](checkpoints.md) | filenames, contents, sizes, and why there is no `vqgan_last.pt` |
| [resume-and-head.md](resume-and-head.md) | `--resume` vs `--vqgan-checkpoint`, adding a refinement head, the three train stages |
| [schedule-units.md](schedule-units.md) | why everything is counted in images, `cosine_lr`, `crossed`, batch-size scaling of the rate |
| [eval-readout.md](eval-readout.md) | the fixed eval subset, codebook metrics, and the four display rows |
| [distributed.md](distributed.md) | DDP: what is split, wrapping order, and the places it is load-bearing |
| [cli-flags.md](cli-flags.md) | how flags are generated from the config dataclasses |
| [doc-refs.md](doc-refs.md) | this scheme, and `scripts/doc_refs.py` |
