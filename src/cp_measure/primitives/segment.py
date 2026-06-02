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
import scipy.ndimage
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


def max_position_per_object(
    values: NDArray[numpy.float64],
    seg0: NDArray[numpy.int64],
    xc: NDArray[numpy.float64],
    yc: NDArray[numpy.float64],
    zc: NDArray[numpy.float64],
    labels_sorted: NDArray[numpy.integer],
) -> tuple[
    NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]
]:
    """Per-object argmax position via ``scipy.ndimage.maximum_position``.

    Reproduces the shipped numpy implementation EXACTLY, including scipy's
    labeled tie-break on exact-equal maxima (an implementation artifact that is
    neither reliably first nor last). The reference calls ``maximum_position``
    once per object on that object's pixels in raster order; this does the same,
    so ``Location_MaxIntensity_*`` is bit-identical to the numpy backend. (We do
    NOT use the numba kernel's ``>=``-last argmax here, by design — exact parity
    with the existing output is the goal.)
    """
    n = int(labels_sorted.size)
    max_x = numpy.zeros(n)
    max_y = numpy.zeros(n)
    max_z = numpy.zeros(n)
    for k in range(n):
        sel = seg0 == k
        vals_k = values[sel]
        if vals_k.size == 0:
            continue
        # Constant labels + single index reproduces the reference's per-object
        # labeled call; the position returned indexes into vals_k's raster order.
        lbl = int(labels_sorted[k])
        labels_k = numpy.full(vals_k.size, lbl, dtype=numpy.int64)
        pos = numpy.array(
            scipy.ndimage.maximum_position(vals_k, labels_k, numpy.array([lbl])),
            dtype=int,
        ).ravel()[0]
        max_x[k] = xc[sel][pos]
        max_y[k] = yc[sel][pos]
        max_z[k] = zc[sel][pos]
    return max_x, max_y, max_z
