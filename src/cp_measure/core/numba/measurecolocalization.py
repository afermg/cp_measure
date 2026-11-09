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
from cp_measure.core.numba._colocalization import coloc_per_object, dense_ranks_grouped
from cp_measure.core.numba._costes import (
    CONSTANT_CHANNEL,
    EMPTY_OBJECT,
    FEW_NONZERO,
    FEW_OR_NONFINITE,
    OUT_OF_RANGE,
    REPLAY,
    RESUME,
    SEARCHED,
    STEPS_DONE,
    STEPS_NOT_CONVERGED,
    ZERO_DENOMINATOR,
    _bisection_end,
    _bisection_steps,
    _coefficients,
    _moment_totals,
    costes_faster,
    costes_linear,
)
from cp_measure.primitives._segment_numba import (
    flatten_pairs_grouped,
    label_counts,
    numpy_sum_chunk,
)
from cp_measure.primitives.parallel import map_images
from cp_measure.primitives.shapes import each_volume, is_batch, memory_key, native

# The reference's warning text for each undefined Costes status.
_UNDEFINED_REASONS = {
    EMPTY_OBJECT: "missing or empty object label",
    FEW_OR_NONFINITE: "fewer than three paired pixels or non-finite signal",
    FEW_NONZERO: "fewer than three nonzero paired pixels",
    CONSTANT_CHANNEL: "constant channel in the regression subset",
    ZERO_DENOMINATOR: "zero above-threshold intensity denominator",
}

# Pixel dtypes the fused kernel reproduces: float64, and float32 (widened exactly; its
# percentage thresholds keep NumPy's float32 rounding, see coloc_per_object).
_KERNEL_DTYPES = (numpy.dtype(numpy.float64), numpy.dtype(numpy.float32))


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
    masks_zyx,
    pixels_1_zyx,
    pixels_2_zyx,
    thr_frac,
    compute_rwc,
    compute_pearson=False,
    threshold=None,
    ranks=(None, None),
    chunk=0,
):
    """Flatten one ``(Z, Y, X)`` image triple and run the fused kernel.

    Returns the nine per-object arrays from ``coloc_per_object``, or ``None`` when
    the image holds no objects (the callers then emit empty feature arrays).
    ``ranks`` are the channels' precomputed RWC ranks (:func:`_channel_ranks`).
    ``chunk`` is ``numpy_sum_chunk()``, read by the caller on its own thread
    (``np.setbufsize`` is thread-local, so worker threads would read the default).
    """
    g1, g2, offsets = _grouped(masks_zyx, pixels_1_zyx, pixels_2_zyx)
    n = offsets.size - 1
    if n == 0:
        return None
    thresholds = None
    if numpy.float32 in (pixels_1_zyx.dtype, pixels_2_zyx.dtype):
        thresholds = _float32_thresholds(
            (g1, g2),
            (pixels_1_zyx.dtype, pixels_2_zyx.dtype),
            offsets,
            threshold,
            thr_frac,
        )
    # Pearson's means use NumPy's pairwise summation (see coloc_per_object).
    return coloc_per_object(
        g1,
        g2,
        offsets,
        n,
        thr_frac,
        compute_rwc,
        compute_pearson,
        chunk,
        thresholds,
        *ranks,
    )


def _channel_ranks(mask, pixels):
    """One channel's RWC ranks over the object blocks of ``mask``, as ``_run`` groups
    them; ``None`` for an image without objects."""
    g, _, offsets = _grouped(mask, pixels, pixels)
    return dense_ranks_grouped(g, offsets) if offsets.size > 1 else None


def _float32_thresholds(grouped, dtypes, offsets, threshold, thr_frac):
    """Per-object percentage thresholds with the reference's dtype rules, shape (2, n).

    A float32 channel gets the reference's own expression, ``(thr / 100) * max``
    evaluated by NumPy on float32 maxima (so it rounds as the reference does for any
    ``thr`` type and NumPy version); a float64 channel keeps the kernel's
    ``thr_frac * max``. Only the threshold, i.e. which pixels count, follows the
    input dtype; the kernel's sums stay float64.
    """
    n = offsets.size - 1
    nonempty = offsets[1:] > offsets[:-1]
    out = numpy.zeros((2, n))
    for row, (values, dtype) in enumerate(zip(grouped, dtypes)):
        maxima = numpy.full(n, numpy.nan)
        if nonempty.any():
            maxima[nonempty] = numpy.maximum.reduceat(values, offsets[:-1][nonempty])
        if dtype == numpy.float32:
            # same expression as reference._percentage_threshold in input mode
            out[row] = (threshold / 100) * maxima.astype(numpy.float32)
        else:
            out[row] = thr_frac * maxima
    return out


def _featurize(
    pixels_1,
    pixels_2,
    masks,
    threshold,
    compute_rwc,
    build,
    ref,
    *,
    force_reference=False,
    **kwargs,
):
    """Split the triple into per-image ``(Z, Y, X)`` views and map each through ``build``.

    ``build(result)`` turns the kernel tuple (or ``None`` for an empty image) into
    one image's feature dict. ``unwrap`` then collapses a single image to that
    dict, or returns the list for a batch — matching the numba intensity backend.
    ``force_reference`` sends every image to ``ref`` (an option the kernel lacks).
    """
    precision = kwargs.get("precision", "input")
    validate_precision(precision)
    # Preserve historical scalar division in input mode, including NumPy 2's
    # float32 rounding. Only the opt-in mode widens before dividing.
    frac = (float(threshold) if precision == "float64" else threshold) / 100.0
    chunk = numpy_sum_chunk()  # on the calling thread: np.setbufsize is per thread
    # Views, no batch copy. each_volume checks that each image's labels and pixels
    # match in shape and rank before unchecked JIT indexing (no mask broadcasting).
    first, unwrap = each_volume(masks, pixels_1)
    second, _ = each_volume(masks, pixels_2)

    def compute(item):
        # Kernel work only (the GIL is released): map_images runs batches on threads.
        m, p1, p2 = item
        ranks = tuple(shared.get((memory_key(m), memory_key(p))) for p in (p1, p2))
        if None in ranks:
            ranks = (None, None)
        # big-endian channels would otherwise miss the kernel dtypes
        p1, p2 = native(p1), native(p2)
        if ref is not reference.get_correlation_pearson:
            p1, p2 = (
                promote_float_pixels(p1, precision),
                promote_float_pixels(p2, precision),
            )
        if (
            force_reference
            or p1.dtype not in _KERNEL_DTYPES
            or p2.dtype not in _KERNEL_DTYPES
        ):
            return m, p1, p2, None
        compute_pearson = ref is reference.get_correlation_pearson
        res = _run(
            m, p1, p2, frac, compute_rwc, compute_pearson, threshold, ranks, chunk
        )
        return m, p1, p2, res

    def finish(staged):
        # Reference fallbacks (and their warnings) on the calling thread, in order.
        m, p1, p2, res = staged
        if (
            force_reference
            or p1.dtype not in _KERNEL_DTYPES
            or p2.dtype not in _KERNEL_DTYPES
        ):
            return ref(p1, p2, m, **kwargs)
        result = build(res)
        if any(not numpy.isfinite(values).all() for values in result.values()):
            result = ref(p1, p2, m, **kwargs)
        return result

    items = [(m, p1, p2) for (m, p1, _), (_, p2, _) in zip(first, second)]
    # RWC's rank sort dominates it. In a batch of channel pairs each channel recurs
    # (once per pair it is in), so rank each (mask, channel) once, on threads, and
    # let every pair read the ranks. Ranks of the exactly widened float64 values
    # equal those of the input, so they are keyed on the caller's arrays.
    shared = {}
    if compute_rwc and not force_reference and len(items) > 1:
        channels = {}
        for m, p1, p2 in items:
            for p in (p1, p2):
                if p.dtype in _KERNEL_DTYPES or precision == "float64":
                    channels.setdefault((memory_key(m), memory_key(p)), (m, p))
        keys = list(channels)
        shared = dict(
            zip(keys, map_images(lambda k: _channel_ranks(*channels[k]), keys))
        )
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
    rank_scope: str = "object",
) -> dict[str, list[float]]:
    """Object-local ranks are accelerated; object-set ranks use the reference."""
    if rank_scope not in ("object", "object_set"):
        raise ValueError("rank_scope must be 'object' or 'object_set'")

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
        force_reference=rank_scope == "object_set",
        thr=thr,
        precision=precision,
        rank_scope=rank_scope,
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


def _resume_bisection(g1, g2, lo, hi, scale, state, chunk):
    """Finish a bisection the kernel paused at an ambiguous step.

    The reference decides each ambiguous step (SciPy's last bits pick the side of
    r = 0, and SciPy emits its own warnings); the kernel runs the steps in between.
    Same values and warnings as replaying ``reference._costes_pair``, with one
    SciPy call per ambiguous step instead of one per step.
    """
    fi, si = g1[lo:hi], g2[lo:hi]
    a, b = float(state[6]), float(state[7])
    totals = _moment_totals(g1, g2, lo, hi)
    code = None
    while code != STEPS_DONE:
        # bisection_costes' update on Python floats, as the reference computes it.
        left, right, mid = (float(v) for v in state[:3])
        thr_fi = mid / scale
        selected = (fi < thr_fi) | (si < (a * thr_fi) + b)
        if numpy.count_nonzero(selected) <= 2:
            left = mid - 1
        else:
            r = reference._costes_background_correlation(fi, si, selected)
            if r < 0:
                left = mid - 1
            elif r >= 0:
                right = mid + 1
                state[4] = mid
            # NaN (constant background) leaves the bounds unchanged, as in CP.
        state[3] = mid
        divisor = 6 / 5 if right - left > 6 else 2
        state[0], state[1], state[2] = left, right, ((right - left) // divisor) + left
        code = _bisection_steps(g1, g2, lo, hi, scale, state, totals, True)
        if code == STEPS_NOT_CONVERGED:
            return reference._undefined_costes("bisection did not converge")
    thr1, thr2 = _bisection_end(scale, state)
    status, c1, c2 = _coefficients(
        g1,
        g2,
        lo,
        hi,
        thr1,
        thr2,
        numpy.empty(hi - lo, bool),
        numpy.empty(hi - lo),
        chunk,
    )
    if status != SEARCHED:
        return reference._undefined_costes(_UNDEFINED_REASONS[status])
    return c1, c2


def get_correlation_costes(
    pixels_1: NDArray[numpy.number],
    pixels_2: NDArray[numpy.number],
    masks: NDArray[numpy.integer],
    fast_costes: str = M_FASTER,
    thr: int = 15,
    *,
    scale_max: int | None = None,
    scipy_ties: bool = False,
) -> dict[str, list[float]] | list[dict[str, list[float]]]:
    """Costes coefficients using the current NumPy scale and warning contracts.

    Floats require an explicit acquisition ``scale_max``; same-dtype unsigned
    integers infer it. ``thr`` is retained but unused. One compiled pass measures
    all objects; its regression and coefficient sums reproduce NumPy's, so the
    same pixels count. Undefined results warn with the reference's messages, and
    near-constant searches replay the reference for its diagnostics. See the NumPy
    API for details.

    The kernel decides every search step itself. When r is within rounding of a
    decision point (often exactly 0 on small integer-grid objects) or the subset is
    nearly constant, the NumPy backend's result depends on SciPy's last bits; there
    the two backends can differ (see the measurement notes). ``scipy_ties=True``
    asks SciPy for those steps, as the NumPy backend does, and also emits SciPy's
    near-constant warnings: the result is then bit-identical to the NumPy backend.
    """
    if fast_costes not in (M_FASTER, M_FAST, M_ACCURATE):
        raise ValueError("Unknown Costes threshold-search mode")
    chunk = numpy_sum_chunk()  # on the calling thread: np.setbufsize is per thread

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
        states = None
        if fast_costes == M_FASTER:
            c1, c2, status, states = costes_faster(
                g1, g2, offsets, float(scale), scipy_ties, chunk
            )
        else:
            accurate = fast_costes == M_ACCURATE
            c1, c2, status = costes_linear(
                g1, g2, offsets, float(scale), accurate, scipy_ties, chunk
            )
        # Keep the grouped pixels only for objects the reference must decide.
        needed = ((status == REPLAY) | (status == RESUME)).any()
        replay = (g1, g2) if needed else (None, None)
        return *replay, offsets, c1, c2, status, states, scale

    def finish(staged):
        g1, g2, offsets, c1, c2, status, states, scale = staged
        # Warnings, errors and replays on the calling thread, in image and object
        # order, as the reference emits them.
        for k in numpy.flatnonzero(status):
            if status[k] == OUT_OF_RANGE:
                raise ValueError(
                    "Costes expects normalized pixels in [0, 1]; check scale_max"
                )
            lo, hi = offsets[k], offsets[k + 1]
            if status[k] == RESUME:
                c1[k], c2[k] = _resume_bisection(
                    g1, g2, lo, hi, scale, states[k], chunk
                )
            elif status[k] == REPLAY:
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
