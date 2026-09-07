"""Running the model over an entire image, and scoring the result.

The model only ever sees `tile_size` x `tile_size` crops, so anything that
wants a *whole* image out of it — scripts/reconstruct.py saving a picture,
scripts/test_whole_image.py measuring one — has to cut the image into tiles,
push those through in batches, and blend the pieces back together. That is one
piece of machinery, so it lives here rather than in each script.

The metrics below are deliberately whole-image metrics. A per-crop L1 says how
well the model reproduces a 256px window; it says nothing about whether the
windows still agree with each other once they are put back side by side, which
is the thing a tiled model can plausibly get wrong.
"""

import torch

from ..data.tiling import plan_tiles, stitch_tiles


@torch.no_grad()
def reconstruct_tiled(
    vqgan, source, tile_size: int, overlap: int, device, batch_size: int = 8, progress=None
) -> torch.Tensor:
    """[3, H, W] in [-1, 1] -> its reconstruction, same size, back on the CPU.

    `plan_tiles` (not the training sampler) lays out the grid: every pixel has
    to be covered exactly once or more for the stitch to fill the frame, which
    is why the last tile in each axis is pulled back to the border instead of
    being left off. `progress` wraps the batch loop if the caller wants a bar.
    """
    _, h, w = source.shape
    if h < tile_size or w < tile_size:
        raise ValueError(f"image is {w}x{h}, smaller than the {tile_size}px tile")

    origins = plan_tiles(h, w, tile_size, overlap)
    batches = range(0, len(origins), batch_size)
    if progress is not None:
        batches = progress(batches)

    recon_tiles = []
    for i in batches:
        chunk = origins[i:i + batch_size]
        batch = torch.stack(
            [source[:, top:top + tile_size, left:left + tile_size] for top, left in chunk]
        ).to(device)
        recon_tiles.append(vqgan(batch).recon.float().cpu())

    return stitch_tiles(torch.cat(recon_tiles), origins, (h, w), overlap)


def psnr(recon: torch.Tensor, original: torch.Tensor, eps: float = 1e-10) -> float:
    """Peak signal-to-noise ratio in dB between two [-1, 1] tensors.

    Measured after mapping to [0, 1], where the peak is 1.0 — that is the scale
    everyone else reports PSNR on, and doing it in [-1, 1] would quietly shift
    every number by 6dB.
    """
    mse = ((recon - original) / 2).pow(2).mean()
    return float(10 * torch.log10(1.0 / mse.clamp_min(eps)))
