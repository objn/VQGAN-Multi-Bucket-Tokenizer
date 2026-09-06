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
"""

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
