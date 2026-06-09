"""Host-side segment helpers (numpy, backend-agnostic).

A labeled image is reduced to a flat *segment* form — ``values[M]`` intensities,
``seg0[M]`` 0-based segment indices, and ``xc/yc/zc[M]`` per-pixel coordinates —
which the segment kernels then reduce. The flattening itself lives in the numba
layer (:func:`cp_measure.primitives._segment_numba.flatten_numba`); this module
holds only the numpy label→index lookup that feeds it.
"""

import numpy
import scipy.ndimage
from numpy.typing import NDArray


def label_to_idx_lut(
    masks: NDArray[numpy.integer],
    *,
    return_bbox: bool = False,
):
    """Build a ``label -> 0..n-1`` lookup over the sorted positive labels.

    Returns ``(lut, n)`` where ``lut[label]`` is the segment index (and ``-1``
    for absent labels / background) and ``n`` is the object count. Output arrays
    are indexed by this segment rank, which equals ``label - 1`` when labels are
    the contiguous ``1..n`` that cp_measure expects.

    Present labels are enumerated with ``scipy.ndimage.find_objects`` (one pass)
    rather than ``numpy.unique`` (a full-image sort): same ascending label set,
    identical LUT, ~3-5x faster on large/3D masks.

    With ``return_bbox=True`` also returns ``origins``, an ``(n, ndim)`` array of each
    object's per-axis bounding-box minimum (the ``find_objects`` slice starts, already
    computed here) — the local-frame origin the moment routines need, for free instead of
    a separate ``numpy.minimum.at`` pass.
    """
    if not numpy.issubdtype(masks.dtype, numpy.integer):
        masks = masks.astype(numpy.intp, copy=False)
    bboxes = scipy.ndimage.find_objects(masks)
    present = [(i + 1, sl) for i, sl in enumerate(bboxes) if sl is not None]
    labels = numpy.array([lab for lab, _ in present], dtype=numpy.int64)
    n = int(labels.size)
    max_label = int(labels[-1]) if n else 0
    lut = numpy.full(max_label + 1, -1, dtype=numpy.int64)
    lut[labels] = numpy.arange(n, dtype=numpy.int64)
    if not return_bbox:
        return lut, n
    origins = numpy.array(
        [[s.start for s in sl] for _, sl in present], dtype=numpy.int64
    ).reshape(n, masks.ndim)
    return lut, n, origins
