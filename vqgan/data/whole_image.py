"""Streaming whole images, for a test pass that scores full reconstructions.

`CropDataset` hands out fixed-size crops, which is what training wants. A test
that reassembles the picture wants the opposite: the image intact, at native
resolution, so it can be tiled, reconstructed and compared against itself. The
shard plumbing (round-robin over DataLoader workers, sequential reads) is the
same; only the unit of work differs.
"""

from torch.utils.data import IterableDataset, get_worker_info

from .crop_dataset import to_tensor


class WholeImageDataset(IterableDataset):
    """Yields [3, H, W] float tensors in [-1, 1], one per image, in shard order.

    Sizes vary from image to image, so there is nothing to collate: run the
    DataLoader with `batch_size=None` and `collate_fn=collate_single`, and let
    each image be batched over its own tiles instead (see
    `vqgan.eval.reconstruct_tiled`). The workers are still worth having — they
    are what keeps JPEG decoding off the main process while the GPU works.

    An image smaller than `min_size` on either side cannot be tiled at native
    resolution, and is yielded as `None` rather than skipped in silence: a test
    result needs to say how much of the split it actually covered, and the
    count is only visible here, inside the worker.
    """

    def __init__(self, shards, *, min_size: int):
        self.shards = list(shards)
        self.min_size = min_size

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        for shard in self.shards[worker_id::num_workers]:
            for image in shard.iter_images():
                w, h = image.size
                if w < self.min_size or h < self.min_size:
                    yield None
                    continue
                yield to_tensor(image)


def collate_single(item):
    """Identity collate for `batch_size=None`.

    The default converter recurses into whatever it is handed; passing images
    of differing shapes (and the `None`s that stand for undersized ones)
    straight through is exactly what this loader wants.
    """
    return item


__all__ = ["WholeImageDataset", "collate_single"]
