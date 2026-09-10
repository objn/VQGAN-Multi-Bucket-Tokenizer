"""Shared console/progress-bar setup for the pipeline scripts.

tqdm.auto and rich.Console both detect whether they're running in a Jupyter
notebook or a plain terminal and render accordingly (ipywidgets HTML bar vs.
ANSI in a terminal; HTML-formatted text vs. ANSI-colored text) — importing
from here instead of `tqdm`/`rich` directly keeps that behavior consistent
across every script without each one re-deriving it.

TrainingDisplay is the training loop's own readout; everything else still
uses the plain `tqdm` bar exported here.
"""

import time

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, ProgressColumn, TextColumn, TimeRemainingColumn
from rich.text import Text
from tqdm.auto import tqdm

console = Console()

# Live values worth watching every few seconds, short label first. The rest of
# what train_step() returns (g_loss, d_loss and the two grad norms) still goes
# to TensorBoard in full — they are what you read when something has already
# gone wrong, not every tick, and each one on screen costs width that the
# fields above use better.
_LOSS_FIELDS = (
    ("recon_loss", "recon", 4),
    ("laplace_loss", "laplace", 4),
    ("lpips_loss", "lpips", 4),
    ("vq_loss", "vq", 4),
    ("d_weight", "d_w", 2),
)


class _RateColumn(ProgressColumn):
    """images/second, in the same unit the schedule is counted in."""

    def render(self, task):
        speed = task.finished_speed or task.speed
        if not speed:
            return Text("     -- img/s", style="progress.data.speed")
        return Text(f"{speed:>8,.1f} img/s", style="progress.data.speed")


class TrainingDisplay:
    """Three fixed rows that redraw in place, instead of one growing line.

    The training loop reports ten-odd numbers; as a single tqdm postfix they
    ran past the width of a tmux pane, and a bar that cannot fit its line
    stops overwriting and starts scrolling, which buries the run's actual log
    (eval results, EMA switch, checkpoints) in redrawn duplicates.

    So: progress on its own row, losses on a second, codebook health on a
    third, each updated on its own cadence and re-measured against the
    terminal width on every refresh — a resized pane re-wraps instead of
    smearing.

        train  27% ---------------- 902,852/3,400,000   72.5 img/s eta 9:33:41
        loss   recon 0.0321  laplace -1.9477  lpips 0.3510  vq 0.0007  d_w 9.31
        code   14,203/16,384 used (86.7%)  perplexity 9,842/16,384 (60.1%)

    Permanent log lines still go through `console.print` as before: rich's
    Live moves this block down and prints above it, so the history stays
    readable and scrollable.

    Off a TTY (`> log.txt`) there is nothing to redraw in place, so the live
    block is skipped entirely and each loss update prints one plain line —
    a log file of one row per log interval rather than thousands of escape
    sequences.
    """

    def __init__(self, *, total_images: int, initial_images: int = 0):
        self.total_images = total_images
        self.completed = initial_images
        self.live_enabled = console.is_terminal
        self._loss_row = Text("loss   waiting for the first interval...", style="dim")
        self._code_row = Text("code   waiting for the first eval...", style="dim")
        self._started_at = time.monotonic()
        self._started_from = initial_images

        self._progress = Progress(
            TextColumn("[bold]train[/bold]"),
            BarColumn(bar_width=None),  # None = take whatever width is left
            TextColumn("{task.completed:,}/{task.total:,}"),
            _RateColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            console=console,
            auto_refresh=False,  # the Live below drives every redraw
        )
        self._task = self._progress.add_task(
            "train", total=total_images, completed=initial_images
        )
        self._live = Live(
            self._render(), console=console, refresh_per_second=4, transient=False
        ) if self.live_enabled else None

    def _render(self):
        return Group(self._progress, self._loss_row, self._code_row)

    def start(self):
        if self._live is not None:
            self._live.start()
        return self

    def stop(self):
        if self._live is not None:
            self._live.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def advance(self, images: int):
        """One training step's worth of images. Cheap enough to call per step:
        the bar's own redraw is rate-limited by Live, not by this."""
        self.completed += images
        self._progress.advance(self._task, images)

    def set_losses(self, logs: dict, lr: float):
        parts = []
        for key, label, places in _LOSS_FIELDS:
            value = logs.get(key)
            if value is None:  # lpips is None when it is switched off
                continue
            parts.append(f"{label} {value:.{places}f}")
        parts.append(f"lr {lr:.2e}")
        body = "  ".join(parts)

        if self.live_enabled:
            self._loss_row = Text.from_markup(f"[bold]loss[/bold]   {body}")
            self._live.update(self._render())
        else:
            # No redraw off a TTY, so fold the progress into the same line —
            # otherwise a log file has losses with no idea where they landed.
            elapsed = max(time.monotonic() - self._started_at, 1e-9)
            rate = (self.completed - self._started_from) / elapsed
            pct = 100.0 * self.completed / max(self.total_images, 1)
            # soft_wrap: one entry per line, however wide. rich otherwise wraps
            # at its assumed 80 columns, which splits every entry in half and
            # makes the file harder to read and to grep than no wrapping at all.
            console.print(
                f"[{self.completed:,}/{self.total_images:,} {pct:5.1f}% "
                f"{rate:,.1f} img/s] {body}",
                soft_wrap=True,
            )

    def set_codebook(self, used: int, total: int, usage_pct: float, perplexity: float):
        """Codebook health, refreshed once per eval — usage and perplexity are
        only measured there, so this row deliberately holds its last values
        between evals rather than blanking out."""
        ratio = 100.0 * perplexity / max(total, 1)
        body = (f"{used:,}/{total:,} used ({usage_pct:.1f}%)  "
                f"perplexity {perplexity:,.0f}/{total:,} ({ratio:.1f}%)")
        if self.live_enabled:
            self._code_row = Text.from_markup(f"[bold]code[/bold]   {body}")
            self._live.update(self._render())


__all__ = ["TrainingDisplay", "console", "tqdm"]
