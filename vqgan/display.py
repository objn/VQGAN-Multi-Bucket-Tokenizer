"""Shared console/progress-bar setup for the pipeline scripts.

tqdm.auto and rich.Console both detect whether they're running in a Jupyter
notebook or a plain terminal and render accordingly (ipywidgets HTML bar vs.
ANSI in a terminal; HTML-formatted text vs. ANSI-colored text) — importing
from here instead of `tqdm`/`rich` directly keeps that behavior consistent
across every script without each one re-deriving it.
"""

from rich.console import Console
from tqdm.auto import tqdm

console = Console()

__all__ = ["console", "tqdm"]
