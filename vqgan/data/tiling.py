"""Cutting a full-resolution image into fixed-size tiles and putting it back.

The model only ever sees `tile` x `tile` crops, so reconstructing a whole
image means running it tile by tile and stitching the results. Two details
make that stitch invisible:

  - The last tile in each axis is *shifted inward* to end exactly at the image
    edge (`tile_origins`) rather than padded, so every tile the model sees is
    full of real pixels. Nothing is ever masked out of a loss, and the model
    never has to learn to ignore a black margin.
  - Tiles are cut with `stride < tile` and blended with a cosine ramp
    (`feather_window`), so neighbouring tiles fade into each other instead of
    meeting at a hard line. Independently decoded tiles disagree slightly at
    their shared boundary; feathering turns that step into a gradient.

Training cuts its tiles differently, with `jittered_tile_origins`: the same
grid, but laid out with a fixed stride and randomly nudged each pass so a
photo does not contribute the identical crops every epoch. Inference keeps
`plan_tiles`, because a reconstruction has to cover every pixel and a jittered
grid deliberately does not.
"""

import random

import torch


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets of `tile`-wide windows covering [0, length).

    The final offset is always `length - tile`, so the last window is pulled
    back inside the image instead of hanging over the edge. That makes its
    overlap with the previous window wider than `tile - stride`, which the
    weight normalization in `stitch_tiles` handles for free.
    """
    if length < tile:
        raise ValueError(f"length {length} is smaller than tile {tile}")
    if not 0 < stride <= tile:
        raise ValueError(f"stride {stride} must be in (0, tile={tile}]")

    origins = list(range(0, length - tile + 1, stride))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


def feather_window(tile: int, overlap: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """[tile, tile] blend weights: a raised-cosine ramp over the outer
    `overlap` pixels on each side, flat 1.0 in the middle.

    Separable, so the 2D window is the outer product of the same 1D ramp.
    Never returns exactly 0 at the border — a tile that is the *only* one
    covering some pixel (which happens at an image corner) must still
    contribute there, and a zero weight would make that pixel 0/0.
    """
    ramp = torch.ones(tile, device=device, dtype=dtype)
    if overlap > 0:
        k = min(overlap, tile // 2)
        edge = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, k + 2, device=device, dtype=dtype)[1:-1]))
        ramp[:k] = edge
        ramp[tile - k:] = edge.flip(0)
    return ramp[:, None] * ramp[None, :]


def stitch_tiles(tiles, origins, out_hw, overlap: int) -> torch.Tensor:
    """Blend `tiles` back into one [C, H, W] image.

    `tiles` is [N, C, tile, tile] and `origins` the matching list of (top, left)
    pairs. Accumulates sum(w * tile) and sum(w) separately and divides at the
    end, so overlap regions are a proper weighted average no matter how many
    tiles cover them (2 in the middle of an edge, 4 at an interior corner, more
    where a shifted-inward last tile piles up).
    """
    n, c, tile, tile_w = tiles.shape
    if tile != tile_w:
        raise ValueError(f"tiles must be square, got {tile}x{tile_w}")
    if len(origins) != n:
        raise ValueError(f"got {n} tiles but {len(origins)} origins")

    h, w = out_hw
    acc = torch.zeros(c, h, w, device=tiles.device, dtype=torch.float32)
    weight = torch.zeros(1, h, w, device=tiles.device, dtype=torch.float32)
    window = feather_window(tile, overlap, device=tiles.device)[None]  # [1, tile, tile]

    for patch, (top, left) in zip(tiles, origins):
        acc[:, top:top + tile, left:left + tile] += patch.float() * window
        weight[:, top:top + tile, left:left + tile] += window

    return acc / weight


def plan_tiles(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """(top, left) origins covering an h x w image, in row-major order."""
    stride = tile - overlap
    return [(t, l) for t in tile_origins(h, tile, stride) for l in tile_origins(w, tile, stride)]


def _axis_origins(length: int, tile: int, stride: int, max_jitter: int, rng, jitter: bool) -> list[int]:
    """Start offsets along one axis: a strided grid, centered, optionally jittered.

    `n` is how many whole tiles fit at this stride. Two cases, and between them
    they cover every image size the dataset accepts:

      - n == 1 (`length` in [tile, tile + stride)): there is no grid to speak
        of, so a training pass takes a *fully* random in-bounds window rather
        than a nudged center one — for a 479px side that is +/-111px of freedom
        instead of +/-19, which is the whole point of cropping a small image.
        With `jitter` off it is the center window, so evaluation is repeatable.
      - n > 1: tiles every `stride` pixels, with the leftover split evenly
        between the two ends instead of piling up at the far edge. Nothing is
        pulled back to touch the border the way `tile_origins` does: the outer
        ~leftover/2 pixels of the image are simply never sampled, which costs
        an edge band but avoids over-sampling one side of every photo.
    """
    if length < tile:
        raise ValueError(f"length {length} is smaller than tile {tile}")

    n = 1 + (length - tile) // stride
    if n == 1:
        return [rng.randint(0, length - tile) if jitter else (length - tile) // 2]

    leftover = length - tile - (n - 1) * stride
    start = leftover // 2
    if jitter:
        # Shift the whole row/column together, and pick the shift from a range
        # that is already in bounds rather than clamping each tile afterwards:
        # clamping would pull an edge tile toward its neighbour and quietly
        # push their overlap past overlap_ratio.
        start += rng.randint(-min(max_jitter, start), min(max_jitter, leftover - start))
    return [start + k * stride for k in range(n)]


def count_tiles(h: int, w: int, tile: int, overlap_ratio: float) -> int:
    """How many crops `jittered_tile_origins` would place over an h x w image.

    Jitter only moves where the tiles land, never how many of them there are,
    so this is the same `n` arithmetic as `_axis_origins` without the RNG —
    cheap enough to run over a whole dataset from image headers alone. 0 if
    the image is smaller than `tile` on either side.
    """
    if h < tile or w < tile:
        return 0
    stride = tile - int(tile * overlap_ratio)
    n_h = 1 + (h - tile) // stride
    n_w = 1 + (w - tile) // stride
    return n_h * n_w


def jittered_tile_origins(
    h: int, w: int, tile: int, overlap_ratio: float, rng: random.Random, *, jitter: bool = True
) -> list[tuple[int, int]]:
    """(top, left) origins for training crops of an h x w image.

    Tiles overlap by `overlap_ratio` of their size, and the grid is nudged by up
    to half that overlap each time, so the crops move from pass to pass. The
    nudge is drawn once per axis and shifts that whole row/column together,
    which keeps the spacing between neighbours exactly `stride` — the overlap
    budget is honoured no matter how the dice land. (Jittering every tile
    independently instead lets two neighbours each step half an overlap *toward*
    each other, doubling the overlap between them, which is the one thing the
    ratio is supposed to bound.)

    `max_jitter` is half the overlap. Deriving it from the stride instead —
    `(stride - (tile - overlap_px)) // 2` — is always 0, since `stride` is
    defined as `tile - overlap_px`.

    Returns origins, not centers, to match `plan_tiles`, `stitch_tiles` and
    PIL's `crop()`; a center is `origin + tile // 2`.
    """
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")

    overlap_px = int(tile * overlap_ratio)
    stride = tile - overlap_px
    max_jitter = overlap_px // 2

    tops = _axis_origins(h, tile, stride, max_jitter, rng, jitter)
    lefts = _axis_origins(w, tile, stride, max_jitter, rng, jitter)
    return [(t, l) for t in tops for l in lefts]
