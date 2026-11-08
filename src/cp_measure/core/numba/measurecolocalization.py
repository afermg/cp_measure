"""Numba-backed colocalization, with current NumPy semantics.

One grouped flatten and a fused per-object kernel handle float64 images.
Percentage-threshold metrics opt into float16/32 promotion with precision="float64".
The default precision="input" retains the pre-policy dtype paths.
Pearson's dtype policy is unchanged. Integer/wider-float inputs use reference
reductions, retaining their existing semantics. Undefined kernel results also use
reference evaluation to preserve warnings and errors, including zero-denominator
Overlap errors.

The public feature schema, label slots and per-object RWC scope are unchanged.
Overlap remains a direct API, not an extra default bulk feature. Costes keeps
NumPy's scale preparation and regression reductions around compiled searches;
no combined featurizer dispatch is introduced here.
"""

import numpy
from numpy.typing import NDArray

from cp_measure._precision import Precision, promote_float_pixels, validate_precision
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
from cp_measure.core.numba._costes import (
    CONSTANT_CHANNEL,
    EMPTY_OBJECT,
    FEW_NONZERO,
    FEW_OR_NONFINITE,
    NO_VALID_THRESHOLD,
    OUT_OF_RANGE,
    REPLAY,
    UNDEFINED_BACKGROUND,
    UNDEFINED_IN_SEARCH,
    ZERO_DENOMINATOR,
    costes_per_object,
)
from cp_measure.primitives._segment_numba import NUMPY_SUM_CHUNK, flatten_pairs_grouped
from cp_measure.primitives.parallel import map_images
from cp_measure.primitives.shapes import to_bzyx

# The reference's warning text for each undefined Costes status.
_UNDEFINED_REASONS = {
    NO_VALID_THRESHOLD: "no finite nonnegative background correlation found",
    UNDEFINED_IN_SEARCH: "background correlation became undefined during linear search",
    UNDEFINED_BACKGROUND: "chosen background subset has undefined correlation",
    EMPTY_OBJECT: "missing or empty object label",
    FEW_OR_NONFINITE: "fewer than three paired pixels or non-finite signal",
    FEW_NONZERO: "fewer than three nonzero paired pixels",
    CONSTANT_CHANNEL: "constant channel in the regression subset",
    ZERO_DENOMINATOR: "zero above-threshold intensity denominator",
}
_COSTES_MODES = {M_FASTER: 0, M_FAST: 1, M_ACCURATE: 2}


def _grouped(masks_zyx, pixels_1_zyx, pixels_2_zyx):
    """Each object's pixels of both channels as contiguous float64 blocks.

    Returns ``(g1, g2, offsets)``: object ``k`` owns ``g1[offsets[k] : offsets[k + 1]]``,
    in raster order like the reference's per-label extraction.
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
    offsets = numpy.zeros(n + 1, dtype=numpy.int64)
    # bincount rejects uint64 even when these bounded label IDs fit intp.
    if masks.dtype == numpy.uint64:
        masks = masks.astype(numpy.intp)
    if n:
        offsets[1:] = numpy.cumsum(numpy.bincount(masks.ravel(), minlength=n + 1)[1:])
    g1, g2 = flatten_pairs_grouped(
        masks,
        numpy.ascontiguousarray(pixels_1_zyx, dtype=numpy.float64),
        numpy.ascontiguousarray(pixels_2_zyx, dtype=numpy.float64),
        offsets,
    )
    return g1, g2, offsets


def _run(
    masks_zyx, pixels_1_zyx, pixels_2_zyx, thr_frac, compute_rwc, compute_pearson=False
):
    """Flatten one ``(Z, Y, X)`` image triple and run the fused kernel.

    Returns the nine per-object arrays from ``coloc_per_object``, or ``None`` when
    the image holds no objects (the callers then emit empty feature arrays).
    """
    g1, g2, offsets = _grouped(masks_zyx, pixels_1_zyx, pixels_2_zyx)
    n = offsets.size - 1
    if n == 0:
        return None
    # Pearson's means use NumPy's pairwise summation (see coloc_per_object).
    return coloc_per_object(
        g1, g2, offsets, n, thr_frac, compute_rwc, compute_pearson, NUMPY_SUM_CHUNK
    )


def _featurize(pixels_1, pixels_2, masks, threshold, compute_rwc, build, ref, **kwargs):
    """Normalise the triple via ``to_bzyx`` and map each image through ``build``.

    ``build(result)`` turns the kernel tuple (or ``None`` for an empty image) into
    one image's feature dict. ``unwrap`` then collapses a single image to that
    dict, or returns the list for a batch — matching the numba intensity backend.
    """
    # to_bzyx promotes 2D and singleton-Z inputs to the same shape and does not
    # validate spatial extents. Check the original triple before unchecked JIT
    # indexing; never broadcast masks or silently discard intensity planes.
    precision = kwargs.get("precision", "input")
    validate_precision(precision)
    shape = numpy.asarray(masks).shape
    if numpy.asarray(pixels_1).shape != shape or numpy.asarray(pixels_2).shape != shape:
        raise ValueError("colocalization images and masks must have the same shape")
    # Preserve historical scalar division in input mode, including NumPy 2's
    # float32 rounding. Only the opt-in mode widens before dividing.
    frac = (float(threshold) if precision == "float64" else threshold) / 100.0
    masks_list, pixels_1_list, unwrap = to_bzyx(masks, pixels_1)
    _, pixels_2_list, _ = to_bzyx(masks, pixels_2)

    def compute(item):
        # Kernel work only (the GIL is released): map_images runs batches on threads.
        m, p1, p2 = item
        if ref is not reference.get_correlation_pearson:
            p1, p2 = (
                promote_float_pixels(p1, precision),
                promote_float_pixels(p2, precision),
            )
        if p1.dtype != numpy.float64 or p2.dtype != numpy.float64:
            return m, p1, p2, None
        compute_pearson = ref is reference.get_correlation_pearson
        return m, p1, p2, _run(m, p1, p2, frac, compute_rwc, compute_pearson)

    def finish(staged):
        # Reference fallbacks (and their warnings) on the calling thread, in order.
        m, p1, p2, res = staged
        if p1.dtype != numpy.float64 or p2.dtype != numpy.float64:
            return ref(p1, p2, m, **kwargs)
        result = build(res)
        if any(not numpy.isfinite(values).all() for values in result.values()):
            result = ref(p1, p2, m, **kwargs)
        return result

    items = list(zip(masks_list, pixels_1_list, pixels_2_list))
    return unwrap([finish(staged) for staged in map_images(compute, items)])


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
    *,
    precision: Precision = "input",
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
        precision=precision,
    )


def get_correlation_rwc(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
    *,
    precision: Precision = "input",
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
        precision=precision,
    )


def get_correlation_overlap(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
    *,
    precision: Precision = "input",
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
        precision=precision,
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

    def compute(item):
        # Kernel work only (the GIL is released): map_images runs batches on threads.
        p1, p2, mask = item
        zyx = (mask, p1, p2) if mask.ndim == 3 else (mask[None], p1[None], p2[None])
        g1, g2, offsets = _grouped(*zyx)
        c1, c2, status = costes_per_object(
            g1, g2, offsets, float(scale), _COSTES_MODES[fast_costes], NUMPY_SUM_CHUNK
        )
        # Keep the grouped pixels only for objects the reference must replay.
        replay = (g1, g2) if (status == REPLAY).any() else (None, None)
        return *replay, offsets, c1, c2, status

    def finish(staged):
        g1, g2, offsets, c1, c2, status = staged
        # Warnings, errors and replays on the calling thread, in image and object
        # order, as the reference emits them.
        for k in numpy.flatnonzero(status):
            if status[k] == OUT_OF_RANGE:
                raise ValueError(
                    "Costes expects normalized pixels in [0, 1]; check scale_max"
                )
            if status[k] == REPLAY:
                lo, hi = offsets[k], offsets[k + 1]
                c1[k], c2[k] = reference._costes_pair(
                    g1[lo:hi], g2[lo:hi], scale, fast_costes
                )
            else:
                reference._undefined_costes(_UNDEFINED_REASONS[status[k]])
        c1, c2 = c1.tolist(), c2.tolist()
        return {f"{F_COSTES_FORMAT}_1": c1, f"{F_COSTES_FORMAT}_2": c2}

    # Single arrays retain their original rank, including singleton Z. Explicit
    # batches follow the existing backend helper; no mask repetition is allowed.
    if not isinstance(masks, (list, tuple)) and labels.ndim < 4:
        return finish(compute((first, second, labels)))
    if isinstance(masks, (list, tuple)):
        if not isinstance(pixels_1, (list, tuple)) or not isinstance(
            pixels_2, (list, tuple)
        ):
            raise ValueError("masks and pixels must both be sequences, or both arrays")
        labels, first, second = list(labels), list(first), list(second)
    masks_list, first_list, unwrap = to_bzyx(labels, first)
    _, second_list, _ = to_bzyx(labels, second)
    items = list(zip(first_list, second_list, masks_list))
    return unwrap([finish(staged) for staged in map_images(compute, items)])
