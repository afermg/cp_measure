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
from cp_measure.core.numba._colocalization import channel_ranks, coloc_image
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
    costes_faster_image,
    costes_linear_image,
)
from cp_measure.primitives._segment_numba import numpy_sum_chunk
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


def _kernel_labels(masks):
    """Contiguous native integer labels, without copying them if they are."""
    # Numba requires native-endian labels; convert values, never reinterpret or
    # byteswap the caller's array in place.
    masks = numpy.ascontiguousarray(masks, dtype=masks.dtype.newbyteorder("="))
    return masks if masks.dtype.kind in "iu" else masks.astype(numpy.intp)


def _kernel_pixels(pixels):
    """Contiguous native float32/float64 pixels, without copying them if they are."""
    dtype = pixels.dtype if pixels.dtype in _KERNEL_DTYPES else numpy.float64
    return numpy.ascontiguousarray(pixels, dtype=dtype)


def _run(
    masks_zyx,
    pixels_1_zyx,
    pixels_2_zyx,
    thr_frac,
    compute_rwc,
    compute_pearson=False,
    frac32=None,
    ranks=(None, None),
    chunk=0,
):
    """Group one ``(Z, Y, X)`` image triple and run the fused kernel, in one call.

    Returns the nine per-object arrays from ``coloc_per_object`` (empty when the
    image holds no objects).
    ``frac32`` is the float32 threshold fraction (:func:`_float32_fraction`), or
    ``None`` when NumPy keeps float32 thresholds in float64.
    ``ranks`` are the channels' precomputed RWC ranks (:func:`_channel_ranks`).
    ``chunk`` is ``numpy_sum_chunk()``, read by the caller on its own thread
    (``np.setbufsize`` is thread-local, so worker threads would read the default).
    """
    # Pearson's means use NumPy's pairwise summation (see coloc_per_object). A
    # Python float holds the float32 fraction exactly.
    return coloc_image(
        _kernel_labels(masks_zyx),
        _kernel_pixels(pixels_1_zyx),
        _kernel_pixels(pixels_2_zyx),
        thr_frac,
        compute_rwc,
        compute_pearson,
        chunk,
        0.0 if frac32 is None else float(frac32),
        frac32 is not None and pixels_1_zyx.dtype == numpy.float32,
        frac32 is not None and pixels_2_zyx.dtype == numpy.float32,
        *ranks,
    )


def _channel_ranks(mask, pixels):
    """One channel's RWC ranks over the object blocks of ``mask``, as ``_run``
    groups them."""
    return channel_ranks(_kernel_labels(mask), _kernel_pixels(pixels))


def _float32_fraction(threshold):
    """The reference's float32 threshold fraction, or ``None``.

    The reference thresholds a float32 channel at ``(thr / 100) * max``, which NumPy
    rounds to float32 unless the scalar type forces float64 (a ``numpy.float64``
    ``thr`` under NumPy 2). The same expression on a float32 one gives the fraction,
    rounded as NumPy rounds it for any ``thr`` type and NumPy version.
    """
    fraction = (threshold / 100) * numpy.ones(1, numpy.float32)
    return fraction[0] if fraction.dtype == numpy.float32 else None


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

    ``build(result)`` picks one image's feature arrays from the kernel tuple. ``unwrap`` then collapses a single image to that
    dict, or returns the list for a batch — matching the numba intensity backend.
    ``force_reference`` sends every image to ``ref`` (an option the kernel lacks).
    """
    precision = kwargs.get("precision", "input")
    validate_precision(precision)
    # Preserve historical scalar division in input mode, including NumPy 2's
    # float32 rounding. Only the opt-in mode widens before dividing.
    frac = (float(threshold) if precision == "float64" else threshold) / 100.0
    frac32 = _float32_fraction(threshold)
    chunk = numpy_sum_chunk()  # on the calling thread: np.setbufsize is per thread
    # Views, no batch copy. each_volume checks that each image's labels and pixels
    # match in shape and rank before unchecked JIT indexing (no mask broadcasting).
    first, unwrap = each_volume(masks, pixels_1)
    second, _ = each_volume(masks, pixels_2)

    def compute(item):
        # One kernel call (the GIL is released): map_images runs batches on threads.
        (m, p1, p2), ranks = item
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
        res = _run(m, p1, p2, frac, compute_rwc, compute_pearson, frac32, ranks, chunk)
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
        arrays = build(res)
        if not all(numpy.isfinite(values).all() for values in arrays.values()):
            return ref(p1, p2, m, **kwargs)
        return arrays

    items = [(m, p1, p2) for (m, p1, _), (_, p2, _) in zip(first, second)]
    ranks = [(None, None)] * len(items)
    # RWC's rank sort dominates it. In a batch of channel pairs each channel recurs
    # (once per pair it is in), so rank each (mask, channel) once, on threads, and
    # let every pair read the ranks. Ranks of the exactly widened float64 values
    # equal those of the input, so they are keyed on the caller's arrays, here on
    # the calling thread.
    if compute_rwc and not force_reference and len(items) > 1:
        keys = [
            [(memory_key(m), memory_key(p)) for p in (p1, p2)] for m, p1, p2 in items
        ]
        channels = {}
        for (m, p1, p2), pair in zip(items, keys):
            for p, key in zip((p1, p2), pair):
                if p.dtype in _KERNEL_DTYPES or precision == "float64":
                    channels.setdefault(key, (m, p))
        unique = list(channels)
        shared = dict(
            zip(
                unique,
                map_images(
                    lambda k: _channel_ranks(*channels[k]),
                    unique,
                    pixels=sum(channels[k][0].size for k in unique),
                ),
            )
        )
        found = [tuple(shared.get(key) for key in pair) for pair in keys]
        ranks = [r if None not in r else (None, None) for r in found]
    staged = map_images(
        compute, list(zip(items, ranks)), pixels=sum(m.size for m, _, _ in items)
    )
    return unwrap([finish(s) for s in staged])


def get_correlation_pearson(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
) -> dict[str, NDArray[numpy.float64]]:
    def build(res):
        return {F_CORRELATION_FORMAT: res[0], F_SLOPE_FORMAT: res[1]}

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
) -> dict[str, NDArray[numpy.float64]]:
    def build(res):
        return {f"{F_MANDERS_FORMAT}_1": res[2], f"{F_MANDERS_FORMAT}_2": res[3]}

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
) -> dict[str, NDArray[numpy.float64]]:
    """Object-local ranks are accelerated; object-set ranks use the reference."""
    if rank_scope not in ("object", "object_set"):
        raise ValueError("rank_scope must be 'object' or 'object_set'")

    def build(res):
        return {f"{F_RWC_FORMAT}_1": res[7], f"{F_RWC_FORMAT}_2": res[8]}

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
) -> dict[str, NDArray[numpy.float64]]:
    def build(res):
        return {
            F_OVERLAP_FORMAT: res[4],
            f"{F_K_FORMAT}_1": res[5],
            f"{F_K_FORMAT}_2": res[6],
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
    Same result as replaying ``reference._costes_pair``, ``(c1, c2, reason)``, with
    one SciPy call per ambiguous step instead of one per step.
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
            return numpy.nan, numpy.nan, "bisection did not converge"
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
        return numpy.nan, numpy.nan, _UNDEFINED_REASONS[status]
    return c1, c2, None


def get_correlation_costes(
    pixels_1: NDArray[numpy.number],
    pixels_2: NDArray[numpy.number],
    masks: NDArray[numpy.integer],
    fast_costes: str = M_FASTER,
    thr: int = 15,
    *,
    scale_max: int | None = None,
    scipy_ties: bool = False,
) -> dict[str, NDArray[numpy.float64]] | list[dict[str, NDArray[numpy.float64]]]:
    """Costes coefficients using the current NumPy scale and warning contracts.

    Floats require an explicit acquisition ``scale_max``; same-dtype unsigned
    integers infer it. ``thr`` is retained but unused. One compiled pass measures
    all objects; its regression and coefficient sums reproduce NumPy's, so the
    same pixels count. Undefined results warn as the reference does, once per reason
    and image with the number of objects, and near-constant searches replay the
    reference for its diagnostics. See the NumPy API for details.

    The kernel decides every search step itself. When r is within rounding of a
    decision point (often exactly 0 on small integer-grid objects) or the subset is
    nearly constant, the NumPy backend's result depends on SciPy's last bits; there
    the two backends can differ (see the measurement notes). ``scipy_ties=True``
    asks SciPy for those steps, as the NumPy backend does, and also emits SciPy's
    near-constant warnings: the result is then bit-identical to the NumPy backend.
    """
    if fast_costes not in (M_FASTER, M_FAST, M_ACCURATE):
        raise ValueError("Unknown Costes threshold-search mode")
    # Fail fast on the calling thread when a float (or signed/mixed-dtype) image is
    # passed without scale_max, e.g. the default featurize() config on a normalized
    # float image. Reuses the reference's dtype/scale check (and its message) on the
    # first image pair, so the error is raised here rather than from a map_images
    # worker mid-batch with a threaded traceback.
    if scale_max is None:
        p1 = pixels_1[0] if is_batch(masks) else pixels_1
        p2 = pixels_2[0] if is_batch(masks) else pixels_2
        reference._costes_scale(numpy.asarray(p1), numpy.asarray(p2), scale_max)
    chunk = numpy_sum_chunk()  # on the calling thread: np.setbufsize is per thread

    def compute(item):
        # Per-image preparation and kernel work (the kernel releases the GIL):
        # map_images runs batches on threads, with no whole-batch float64 copy.
        p1, p2, mask = item
        first, second = numpy.asarray(p1), numpy.asarray(p2)
        scale = reference._costes_scale(first, second, scale_max)
        # The kernel widens float32/float64 pixels itself; others convert here.
        first, second = (
            p if p.dtype in _KERNEL_DTYPES else reference._costes_float64(p, scale)
            for p in (first, second)
        )
        labels = numpy.asarray(mask)
        if labels.shape != first.shape or labels.dtype.kind not in "iu":
            raise ValueError(
                "Costes labels must be an integer array matching the images"
            )
        if labels.ndim not in (2, 3):
            raise ValueError(f"Costes expects 2D or 3D images, got {labels.ndim}D")
        if labels.ndim != 3:
            labels, first, second = labels[None], first[None], second[None]
        zyx = (_kernel_labels(labels), _kernel_pixels(first), _kernel_pixels(second))
        # One kernel call; the grouped pixels come back only for replays.
        states = None
        if fast_costes == M_FASTER:
            c1, c2, status, states, g1, g2, offsets = costes_faster_image(
                *zyx, float(scale), scipy_ties, chunk
            )
        else:
            accurate = fast_costes == M_ACCURATE
            c1, c2, status, g1, g2, offsets = costes_linear_image(
                *zyx, float(scale), accurate, scipy_ties, chunk
            )
        return g1, g2, offsets, c1, c2, status, states, scale

    def finish(staged):
        g1, g2, offsets, c1, c2, status, states, scale = staged
        # Replays, errors and warnings on the calling thread, in image and object
        # order, as the reference emits them. Only flagged objects need Python.
        flagged = numpy.flatnonzero(status)
        codes = status[flagged]
        undefined = []  # (first object, reason, number of objects)
        replays = (codes == RESUME) | (codes == REPLAY)
        for k, code in zip(flagged[replays].tolist(), codes[replays].tolist()):
            lo, hi = offsets[k], offsets[k + 1]
            if code == RESUME:
                c1[k], c2[k], reason = _resume_bisection(
                    g1, g2, lo, hi, scale, states[k], chunk
                )
            else:
                c1[k], c2[k], reason = reference._costes_pair(
                    g1[lo:hi], g2[lo:hi], scale, fast_costes
                )
            if reason is not None:
                undefined.append((k, reason, 1))
        # The kernel stops at the first out-of-range object, so it is the last one.
        if codes.size and codes[-1] == OUT_OF_RANGE:
            raise ValueError(
                "Costes expects normalized pixels in [0, 1]; check scale_max"
            )
        found, first, count = numpy.unique(
            codes[~replays], return_index=True, return_counts=True
        )
        others = flagged[~replays]
        for code, i, n in zip(found.tolist(), first.tolist(), count.tolist()):
            undefined.append((others[i], _UNDEFINED_REASONS[code], n))
        counts = {}  # in the order the reasons first occur, as the reference
        for _, reason, n in sorted(undefined, key=lambda entry: entry[0]):
            counts[reason] = counts.get(reason, 0) + n
        reference._warn_undefined_costes(counts)
        return {f"{F_COSTES_FORMAT}_1": c1, f"{F_COSTES_FORMAT}_2": c2}

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
    staged = map_images(compute, items, pixels=sum(m.size for _, _, m in items))
    return [finish(s) for s in staged]
