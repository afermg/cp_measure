"""Simple synthetic images for benchmarking.

``generate`` returns a label mask of ``n_objects`` axis-aligned ellipses on a regular grid
(contiguous ``1..n``) plus intensity channels made of a few big Gaussian blobs at random
positions (re-randomised per channel). Deliberately minimal — just enough to exercise the
measurement functions.
"""

from __future__ import annotations

import numpy
from numpy.typing import NDArray

__version__ = "0.3.0"

_BLOBS_PER_CHANNEL = 5


def generate(
    image_size: int, n_objects: int, n_channels: int = 2, seed: int = 0
) -> tuple[NDArray[numpy.int32], NDArray[numpy.float32]]:
    rng = numpy.random.default_rng(seed)
    labels = _grid_ellipses(image_size, n_objects)
    channels = numpy.stack([_blobs(rng, image_size) for _ in range(n_channels)])
    return labels, channels


def _grid_ellipses(size: int, n: int) -> NDArray[numpy.int32]:
    """``n`` ellipses laid out on a ``rows × cols`` grid, labelled ``1..n``."""
    labels = numpy.zeros((size, size), numpy.int32)
    if n == 0:
        return labels
    cols = int(numpy.ceil(numpy.sqrt(n)))
    rows = int(numpy.ceil(n / cols))
    a, b = (
        0.35 * size / rows,
        0.35 * size / cols,
    )  # semi-axes, leaving a gap between cells
    yy, xx = numpy.mgrid[0:size, 0:size]
    for k in range(n):
        r, c = divmod(k, cols)
        cy, cx = (r + 0.5) * size / rows, (c + 0.5) * size / cols
        labels[((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2 <= 1] = k + 1
    return labels


def _blobs(rng: numpy.random.Generator, size: int) -> NDArray[numpy.float32]:
    """A few big Gaussian blobs at random positions."""
    yy, xx = numpy.mgrid[0:size, 0:size]
    img = numpy.zeros((size, size))
    for _ in range(_BLOBS_PER_CHANNEL):
        cy, cx = rng.uniform(0, size, 2)
        sigma = rng.uniform(size / 10, size / 5)
        img += numpy.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    return img.astype(numpy.float32)
