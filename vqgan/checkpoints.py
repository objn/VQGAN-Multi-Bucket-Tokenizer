"""Finding checkpoints on disk by their step numbers.

Every checkpoint this project writes is named for the training it holds, and
no two are ever named the same, because the counters in the name only move
forward across a model's whole life:

    vqgan_step0028582.pt                      a model with no refinement head
    vqgan_step0028582_refine_step0050000.pt   the same model, head at 50k steps

There is deliberately no `vqgan_last.pt` or any other fixed name: a fixed name
is a file that every run overwrites, and what it overwrites is whatever the run
before it produced. So "the newest checkpoint" is not a name to open, it is a
question to ask the directory, and this is where it gets asked.

Kept out of main.py because the menu is not the only caller — evaluate,
reconstruct, test_whole_image and visualize_model all need the same answer for
their --vqgan-checkpoint default, and a CLI script importing the interactive
menu to get it would be backwards.
"""

import re
from pathlib import Path

# The refine half is optional, and its absence is meaningful: no second number
# means no head in the file.
_STEP_CKPT_RE = re.compile(r"vqgan_step(\d+)(?:_refine_step(\d+))?\.pt")


def checkpoint_steps(name):
    """-> (step, refine_step) for a checkpoint filename, or None if the name is
    not one. `refine_step` is 0 for a model with no head."""
    m = _STEP_CKPT_RE.fullmatch(str(name))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


def latest_step_checkpoint(checkpoint_dir) -> str:
    """The newest checkpoint in `checkpoint_dir`, or "" if it holds none.

    Ordered by the pair (step, refine_step), which is a total order along any
    one model's history because neither counter ever decreases: a refine_only
    stretch leaves the first number alone and separates its files by the
    second, and once the backbone is training again the first takes over. Two
    unrelated lineages sharing a directory can still tie or cross — that is a
    directory to split with --checkpoint-dir, not something a sort can fix.

    Sorted on the parsed numbers rather than on the filename, since the zero
    padding that makes those agree today would stop agreeing for a run that
    outgrew seven digits, and not on mtime, which says when a file was last
    copied about rather than what is in it.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return ""
    candidates = []
    for p in checkpoint_dir.iterdir():
        steps = checkpoint_steps(p.name)
        if steps is not None:
            candidates.append((steps, p))
    if not candidates:
        return ""
    return str(max(candidates, key=lambda t: t[0])[1])


def default_checkpoint(checkpoint_dir) -> str:
    """`latest_step_checkpoint()` for a script's --vqgan-checkpoint default.

    Falls back to a `vqgan_step*.pt` path that does not exist rather than to
    "", so a run in an empty directory fails with a missing-file error naming
    the place it looked instead of an empty-path one that names nothing.
    """
    return latest_step_checkpoint(checkpoint_dir) or str(
        Path(checkpoint_dir) / "vqgan_step0000000.pt"
    )
