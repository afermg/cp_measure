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
NumPy's scale preparation and measures all objects in one compiled pass whose
sums reproduce NumPy's; no combined featurizer dispatch is introduced here.
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
    costes_faster,
    costes_linear,
)
from cp_measure.primitives._segment_numba import (
    flatten_pairs_grouped,
    label_counts,
    numpy_sum_chunk,
)
from cp_measure.primitives.parallel import map_images
from cp_measure.primitives.shapes import each_volume, is_batch

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
    # per-label pixel counts, with no lookup table.
    n = int(masks.max(initial=0))
    offsets = numpy.zeros(n + 1, dtype=numpy.int64)
    if n:
        offsets[1:] = numpy.cumsum(label_counts(masks, n))
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
    """Split the triple into per-image ``(Z, Y, X)`` views and map each through ``build``.

    ``build(result)`` turns the kernel tuple (or ``None`` for an empty image) into
    one image's feature dict. ``unwrap`` then collapses a single image to that
    dict, or returns the list for a batch — matching the numba intensity backend.
    """
    precision = kwargs.get("precision", "input")
    validate_precision(precision)
    # Preserve historical scalar division in input mode, including NumPy 2's
    # float32 rounding. Only the opt-in mode widens before dividing.
    frac = (float(threshold) if precision == "float64" else threshold) / 100.0
    # Views, no batch copy. each_volume checks that each image's labels and pixels
    # match in shape and rank before unchecked JIT indexing (no mask broadcasting).
    first, unwrap = each_volume(masks, pixels_1)
    second, _ = each_volume(masks, pixels_2)

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

    items = [(m, p1, p2) for (m, p1, _), (_, p2, _) in zip(first, second)]
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
    integers infer it. ``thr`` is retained but unused. One compiled pass measures
    all objects; its regression and coefficient sums reproduce NumPy's, so the
    same pixels count. Undefined results warn with the reference's messages, and
    near-constant searches replay the reference for its diagnostics. See the NumPy
    API for details.
    """
    if fast_costes not in (M_FASTER, M_FAST, M_ACCURATE):
        raise ValueError("Unknown Costes threshold-search mode")

    def compute(item):
        # Per-image preparation and kernel work (the kernel releases the GIL):
        # map_images runs batches on threads, with no whole-batch float64 copy.
        p1, p2, mask = item
        first, second, scale = reference._prepare_costes_inputs(p1, p2, scale_max)
        labels = numpy.asarray(mask)
        if labels.shape != first.shape or not numpy.issubdtype(
            labels.dtype, numpy.integer
        ):
            raise ValueError(
                "Costes labels must be an integer array matching the images"
            )
        if labels.ndim not in (2, 3):
            raise ValueError(f"Costes expects 2D or 3D images, got {labels.ndim}D")
        zyx = (labels, first, second)
        if labels.ndim != 3:
            zyx = (labels[None], first[None], second[None])
        g1, g2, offsets = _grouped(*zyx)
        if fast_costes == M_FASTER:
            c1, c2, status = costes_faster(
                g1, g2, offsets, float(scale), numpy_sum_chunk()
            )
        else:
            accurate = fast_costes == M_ACCURATE
            c1, c2, status = costes_linear(
                g1, g2, offsets, float(scale), accurate, numpy_sum_chunk()
            )
        # Keep the grouped pixels only for objects the reference must replay.
        replay = (g1, g2) if (status == REPLAY).any() else (None, None)
        return *replay, offsets, c1, c2, status, scale

    def finish(staged):
        g1, g2, offsets, c1, c2, status, scale = staged
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
        return {
            f"{F_COSTES_FORMAT}_1": c1.tolist(),
            f"{F_COSTES_FORMAT}_2": c2.tolist(),
        }

    # Single arrays keep their original rank, including singleton Z.
    if not is_batch(masks):
        return finish(compute((pixels_1, pixels_2, masks)))
    first, _ = each_volume(masks, pixels_1)
    second, _ = each_volume(masks, pixels_2)
    items = [
        (
            p1 if ndim == 3 else p1[0],
            p2 if ndim == 3 else p2[0],
            m if ndim == 3 else m[0],
        )
        for (m, p1, ndim), (_, p2, _) in zip(first, second)
    ]
    return [finish(staged) for staged in map_images(compute, items)]
