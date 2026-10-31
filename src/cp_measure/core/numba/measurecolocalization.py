"""Numba-backed colocalization, with current NumPy semantics.

One grouped flatten and a fused per-object kernel handle float64 images.
Other dtypes use the reference reductions explicitly: promoting float32 changes
threshold inclusion (pending #124), and integer products can overflow in the
reference. Undefined kernel results also use the reference to preserve warnings
and errors, including the existing zero-denominator Overlap error.

The public feature schema, label slots and per-object RWC scope are unchanged.
Overlap remains a direct API, not an extra default bulk feature. Costes keeps
NumPy's scale preparation and regression reductions around compiled searches;
no combined featurizer dispatch is introduced here.
"""

import numpy
from numpy.typing import NDArray

from cp_measure.core import measurecolocalization as reference
from cp_measure.core.measurecolocalization import (
    F_CORRELATION_FORMAT,
    F_COSTES_FORMAT,
    F_K_FORMAT,
    F_MANDERS_FORMAT,
    F_OVERLAP_FORMAT,
    F_RWC_FORMAT,
    F_SLOPE_FORMAT,
    M_ACCURATE,
    M_FAST,
    M_FASTER,
)
from cp_measure.core.numba._colocalization import coloc_per_object
from cp_measure.core.numba._costes import _bisection, _linear, _regression_ab
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


def _costes_pair(fi, si, scale, mode):
    """Share canonicalization, reductions and diagnostics with the reference."""
    fi = reference._canonical_costes_pixels(fi, scale)
    si = reference._canonical_costes_pixels(si, scale)
    if not reference._valid_costes_pixels(fi, si):
        return numpy.nan, numpy.nan
    a, b = _regression_ab(fi, si, 0, fi.size)
    if not (numpy.isfinite(a) and numpy.isfinite(b)):
        return reference._costes_pair(fi, si, scale, mode)
    if mode == M_FASTER:
        first, second = _bisection(fi, si, 0, fi.size, a, b, scale)
    else:
        first, second = _linear(fi, si, 0, fi.size, a, b, scale, mode == M_ACCURATE)
    if not (numpy.isfinite(first) and numpy.isfinite(second)):
        # The kernel does not emit Python warnings. Replay only exceptional
        # objects to preserve the reference's precise warning categories/messages.
        return reference._costes_pair(fi, si, scale, mode)
    combined = (fi > first) & (si > second)
    total_first = fi[fi >= first].sum()
    total_second = si[si >= second].sum()
    if total_first <= 0 or total_second <= 0:
        return reference._undefined_costes("zero above-threshold intensity denominator")
    return float(fi[combined].sum() / total_first), float(
        si[combined].sum() / total_second
    )


def get_correlation_costes(
    pixels_1: NDArray[numpy.number],
    pixels_2: NDArray[numpy.number],
    masks: NDArray[numpy.integer],
    fast_costes: str = M_FASTER,
    thr: int = 15,
    *,
    scale_max: int | None = None,
) -> dict[str, list[float]]:
    """Costes coefficients using the current NumPy scale and warning contracts.

    Floats require an explicit acquisition ``scale_max``; same-dtype unsigned
    integers infer it. ``thr`` is retained but unused. The repeated threshold
    search is compiled; regression and coefficient reductions remain NumPy to
    preserve threshold inclusion. Undefined/near-constant searches use the
    reference implementation for diagnostics. See the NumPy API for details.
    """
    # Share acquisition-scale preparation; _costes_pair canonicalizes each object.
    first, second, scale = reference._prepare_costes_inputs(
        pixels_1, pixels_2, scale_max
    )
    labels = numpy.asarray(masks)
    if labels.shape != first.shape or not numpy.issubdtype(labels.dtype, numpy.integer):
        raise ValueError("Costes labels must be an integer array matching the images")
    if fast_costes not in (M_FASTER, M_FAST, M_ACCURATE):
        raise ValueError("Unknown Costes threshold-search mode")

    def measure(p1, p2, mask):
        c1, c2 = [], []
        for fi, si in reference._iter_label_pixels(p1, p2, mask):
            if fi is None or si is None or fi.size == 0:
                v1, v2 = reference._undefined_costes("missing or empty object label")
            else:
                v1, v2 = _costes_pair(fi, si, scale, fast_costes)
            c1.append(v1)
            c2.append(v2)
        return {f"{F_COSTES_FORMAT}_1": c1, f"{F_COSTES_FORMAT}_2": c2}

    # Single arrays retain their original rank, including singleton Z. Explicit
    # batches follow the existing backend helper; no mask repetition is allowed.
    if not isinstance(masks, (list, tuple)) and labels.ndim < 4:
        return measure(first, second, labels)
    if isinstance(masks, (list, tuple)):
        if not isinstance(pixels_1, (list, tuple)) or not isinstance(
            pixels_2, (list, tuple)
        ):
            raise ValueError("masks and pixels must both be sequences, or both arrays")
        labels, first, second = list(labels), list(first), list(second)
    masks_list, first_list, unwrap = to_bzyx(labels, first)
    _, second_list, _ = to_bzyx(labels, second)
    return unwrap(
        [measure(p1, p2, m) for m, p1, p2 in zip(masks_list, first_list, second_list)]
    )
