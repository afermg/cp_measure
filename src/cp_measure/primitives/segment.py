"""Host-side segment helpers (numpy/scipy, backend-agnostic).

A labeled image is reduced to a flat *segment* form:

* ``values[M]``  finite labeled pixel intensities (float64)
* ``seg0[M]``    0-based segment index = rank of the pixel's label among the
                 sorted positive labels (``label_to_idx``)
* ``xc/yc/zc[M]``per-pixel coordinates (z = 0 plane for 2D input)

Segment kernels (numpy or numba) then loop over these flat M-length arrays with
no notion of image shape or batch axis. 2D vs 3D differ only in which coordinate
arrays are populated; a batch differs only in how ``seg0`` offsets are assigned.
"""

import numpy
from numpy.typing import NDArray


def label_to_idx_lut(
    masks: NDArray[numpy.integer],
) -> tuple[NDArray[numpy.int64], NDArray[numpy.integer], int]:
    """Build a ``label -> 0..n-1`` lookup over the sorted positive labels.

    Returns ``(lut, labels_sorted, n)`` where ``lut[label]`` is the segment
    index (and ``-1`` for absent labels / background), ``labels_sorted`` are the
    ascending positive labels, and ``n`` is the object count. Output arrays are
    indexed by this segment rank, which equals ``label - 1`` when labels are the
    contiguous ``1..n`` that cp_measure expects.
    """
    unique = numpy.unique(masks)
    labels = unique[unique > 0]
    n = int(labels.size)
    max_label = int(labels[-1]) if n else 0
    lut = numpy.full(max_label + 1, -1, dtype=numpy.int64)
    lut[labels] = numpy.arange(n, dtype=numpy.int64)
    return lut, labels, n


def flatten_labeled(
    masks: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating],
    lut: NDArray[numpy.int64],
) -> tuple[
    NDArray[numpy.float64],
    NDArray[numpy.int64],
    NDArray[numpy.float64],
    NDArray[numpy.float64],
    NDArray[numpy.float64],
]:
    """Flatten a labeled (Z, Y, X) image to ``(values, seg0, xc, yc, zc)``.

    Non-finite pixels and background are dropped, matching the numpy reference's
    ``(masks > 0) & isfinite(pixels)`` mask. Coordinates use ``numpy.mgrid`` over
    the full volume exactly as the reference does, so positions line up.
    """
    lmask = (masks > 0) & numpy.isfinite(pixels)
    values = pixels[lmask].astype(numpy.float64)
    seg0 = lut[masks[lmask]]
    mesh_z, mesh_y, mesh_x = numpy.mgrid[
        0 : masks.shape[0], 0 : masks.shape[1], 0 : masks.shape[2]
    ]
    xc = mesh_x[lmask].astype(numpy.float64)
    yc = mesh_y[lmask].astype(numpy.float64)
    zc = mesh_z[lmask].astype(numpy.float64)
    return values, seg0, xc, yc, zc
