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
) -> tuple[NDArray[numpy.int64], int]:
    """Build a ``label -> 0..n-1`` lookup over the sorted positive labels.

    Returns ``(lut, n)`` where ``lut[label]`` is the segment index (and ``-1``
    for absent labels / background) and ``n`` is the object count. Output arrays
    are indexed by this segment rank, which equals ``label - 1`` when labels are
    the contiguous ``1..n`` that cp_measure expects.

    Present labels are enumerated with ``scipy.ndimage.find_objects`` (one pass)
    rather than ``numpy.unique`` (a full-image sort): same ascending label set,
    identical LUT, ~3-5x faster on large/3D masks.
    """
    if not numpy.issubdtype(masks.dtype, numpy.integer):
        masks = masks.astype(numpy.intp, copy=False)
    bboxes = scipy.ndimage.find_objects(masks)
    labels = numpy.array(
        [i + 1 for i, sl in enumerate(bboxes) if sl is not None], dtype=numpy.int64
    )
    n = int(labels.size)
    max_label = int(labels[-1]) if n else 0
    lut = numpy.full(max_label + 1, -1, dtype=numpy.int64)
    lut[labels] = numpy.arange(n, dtype=numpy.int64)
    return lut, n
