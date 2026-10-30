"""Numba-backed non-Costes colocalization, with current NumPy semantics.

One grouped flatten and a fused per-object kernel handle float64 images.
Other dtypes use the reference reductions explicitly: promoting float32 changes
threshold inclusion (pending #124), and integer products can overflow in the
reference. Undefined kernel results also use the reference to preserve warnings
and errors, including the existing zero-denominator Overlap error.

The public feature schema, label slots and per-object RWC scope are unchanged.
Overlap remains a direct API, not an extra default bulk feature. Costes is owned
by its stacked follow-up; no combined featurizer dispatch is introduced here.
"""

import numpy
from numpy.typing import NDArray

from cp_measure.core import measurecolocalization as reference
from cp_measure.core.measurecolocalization import (
    F_CORRELATION_FORMAT,
    F_K_FORMAT,
    F_MANDERS_FORMAT,
    F_OVERLAP_FORMAT,
    F_RWC_FORMAT,
    F_SLOPE_FORMAT,
)
from cp_measure.core.numba._colocalization import coloc_per_object
from cp_measure.primitives._segment_numba import flatten_pairs_grouped
from cp_measure.primitives.shapes import to_bzyx


def _run(
    masks_zyx, pixels_1_zyx, pixels_2_zyx, thr_frac, compute_rwc, compute_pearson=False
):
    """Flatten one ``(Z, Y, X)`` image triple and run the fused kernel.

    Returns the nine per-object arrays from ``coloc_per_object``, or ``None`` when
    the image holds no objects (the callers then emit empty feature arrays).
    """
    # Numba requires native-endian labels; convert values, never reinterpret or
    # byteswap the caller's array in place. Native contiguous inputs stay zero-copy.
    masks = numpy.ascontiguousarray(masks_zyx, dtype=masks_zyx.dtype.newbyteorder("="))
    if not numpy.issubdtype(masks.dtype, numpy.integer):
        masks = masks.astype(numpy.intp)
    # Labels are contiguous 1..N by contract (see cp_measure._sanitize), so the
    # segment index is label - 1 and the CSR block bounds are the cumulative
    # per-label pixel counts — one bincount, no lookup table.
    n = int(masks.max(initial=0))
    if n == 0:
        return None
    offsets = numpy.zeros(n + 1, dtype=numpy.int64)
    # bincount rejects uint64 even when these bounded label IDs fit intp.
    if masks.dtype == numpy.uint64:
        masks = masks.astype(numpy.intp)
    offsets[1:] = numpy.cumsum(numpy.bincount(masks.ravel(), minlength=n + 1)[1:])
    g1, g2 = flatten_pairs_grouped(
        masks,
        numpy.ascontiguousarray(pixels_1_zyx, dtype=numpy.float64),
        numpy.ascontiguousarray(pixels_2_zyx, dtype=numpy.float64),
        offsets,
    )
    means = None
    if compute_pearson:
        # NumPy's pairwise mean reduction matters for constant and high-offset
        # signals: a serial JIT sum can turn an undefined correlation finite.
        # Reuse exactly that reduction, without changing the other fused metrics.
        means = numpy.zeros((2, n))
        for k in range(n):
            lo, hi = offsets[k : k + 2]
            if hi > lo:
                means[0, k] = numpy.mean(g1[lo:hi])
                means[1, k] = numpy.mean(g2[lo:hi])
    return coloc_per_object(g1, g2, offsets, n, thr_frac, compute_rwc, means)


def _featurize(pixels_1, pixels_2, masks, threshold, compute_rwc, build, ref, **kwargs):
    """Normalise the triple via ``to_bzyx`` and map each image through ``build``.

    ``build(result)`` turns the kernel tuple (or ``None`` for an empty image) into
    one image's feature dict. ``unwrap`` then collapses a single image to that
    dict, or returns the list for a batch — matching the numba intensity backend.
    """
    # to_bzyx promotes 2D and singleton-Z inputs to the same shape and does not
    # validate spatial extents. Check the original triple before unchecked JIT
    # indexing; never broadcast masks or silently discard intensity planes.
    shape = numpy.asarray(masks).shape
    if numpy.asarray(pixels_1).shape != shape or numpy.asarray(pixels_2).shape != shape:
        raise ValueError("colocalization images and masks must have the same shape")
    frac = threshold / 100.0
    masks_list, pixels_1_list, unwrap = to_bzyx(masks, pixels_1)
    _, pixels_2_list, _ = to_bzyx(masks, pixels_2)
    results = []
    for m, p1, p2 in zip(masks_list, pixels_1_list, pixels_2_list):
        if p1.dtype != numpy.float64 or p2.dtype != numpy.float64:
            result = ref(p1, p2, m, **kwargs)
        else:
            result = build(
                _run(
                    m,
                    p1,
                    p2,
                    frac,
                    compute_rwc,
                    compute_pearson=ref is reference.get_correlation_pearson,
                )
            )
            if any(not numpy.isfinite(values).all() for values in result.values()):
                result = ref(p1, p2, m, **kwargs)
        results.append(result)
    return unwrap(results)


def get_correlation_pearson(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
) -> dict[str, list[float]]:
    def build(res):
        if res is None:
            return {F_CORRELATION_FORMAT: [], F_SLOPE_FORMAT: []}
        corr, slope = res[0].tolist(), res[1].tolist()
        return {F_CORRELATION_FORMAT: corr, F_SLOPE_FORMAT: slope}

    return _featurize(
        pixels_1, pixels_2, masks, 15, False, build, reference.get_correlation_pearson
    )


def get_correlation_manders_fold(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, list[float]]:
    def build(res):
        if res is None:
            return {f"{F_MANDERS_FORMAT}_1": [], f"{F_MANDERS_FORMAT}_2": []}
        m1, m2 = res[2].tolist(), res[3].tolist()
        return {f"{F_MANDERS_FORMAT}_1": m1, f"{F_MANDERS_FORMAT}_2": m2}

    return _featurize(
        pixels_1,
        pixels_2,
        masks,
        thr,
        False,
        build,
        reference.get_correlation_manders_fold,
        thr=thr,
    )


def get_correlation_rwc(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, list[float]]:
    def build(res):
        if res is None:
            return {f"{F_RWC_FORMAT}_1": [], f"{F_RWC_FORMAT}_2": []}
        rwc1, rwc2 = res[7].tolist(), res[8].tolist()
        return {f"{F_RWC_FORMAT}_1": rwc1, f"{F_RWC_FORMAT}_2": rwc2}

    return _featurize(
        pixels_1,
        pixels_2,
        masks,
        thr,
        True,
        build,
        reference.get_correlation_rwc,
        thr=thr,
    )


def get_correlation_overlap(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, list[float]]:
    def build(res):
        if res is None:
            return {
                F_OVERLAP_FORMAT: [],
                f"{F_K_FORMAT}_1": [],
                f"{F_K_FORMAT}_2": [],
            }
        overlap, k1, k2 = res[4].tolist(), res[5].tolist(), res[6].tolist()
        return {
            F_OVERLAP_FORMAT: overlap,
            f"{F_K_FORMAT}_1": k1,
            f"{F_K_FORMAT}_2": k2,
        }

    return _featurize(
        pixels_1,
        pixels_2,
        masks,
        thr,
        False,
        build,
        reference.get_correlation_overlap,
        thr=thr,
    )
