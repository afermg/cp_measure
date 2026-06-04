"""Numba-backed MeasureColocalization (Pearson, Manders, RWC, Overlap).

Drop-in replacements for the like-named functions in
:mod:`cp_measure.core.measurecolocalization`, producing the identical per-object
feature dicts. The reference iterates ``labels_to_binmasks`` — materialising an
``(N, H, W)`` boolean stack — and re-indexes the whole image per object through
``scipy.ndimage`` (single-label reductions, hence the ``[0]`` everywhere). Here a
single grouped flatten plus one fused per-object kernel
(:func:`cp_measure.core.numba._colocalization.coloc_per_object`) replaces all of
it.

Input is normalised through :func:`cp_measure.primitives.shapes.to_bzyx`, exactly
like the numba ``intensity``/``zernike`` backends. Colocalization takes a
``(pixels_1, pixels_2, masks)`` triple, so ``to_bzyx`` is called twice against the
shared mask and the single ``unwrap`` is reused. Every feature here is a function
of the per-object value vectors ONLY (no pixel coordinates), so the kernels never
branch on 2D vs 3D and the ``(1, Y, X)`` vs ``(H, W)`` divergence that affects the
intensity backend cannot occur.

Pixels are upcast to float64 before reduction. This is strictly more accurate
than the numpy reference on integer-dtype input — where ``fi*si`` overflows in
uint8 (Overlap/K can exceed 1) and the Pearson slope's ``lstsq`` runs in float32
— but means the two backends can differ on genuine integer images. Real (float)
intensity images are unaffected.

``costes`` is also here (it runs a per-object iterative threshold search rather
than the fused reduction, so it has its own kernel and runner).
"""

import numpy
from numpy.typing import NDArray

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
    infer_scale,
)
from cp_measure.core.numba._colocalization import coloc_per_object
from cp_measure.core.numba._costes import costes_per_object
from cp_measure.primitives._segment_numba import flatten_pairs_grouped
from cp_measure.primitives.segment import labels_to_offsets
from cp_measure.primitives.shapes import to_bzyx

_COSTES_MODE = {M_FASTER: 0, M_FAST: 1, M_ACCURATE: 2}


def _flatten_image(masks_zyx, pixels_1_zyx, pixels_2_zyx):
    """Normalise one ``(Z, Y, X)`` triple to grouped per-object value blocks.

    Returns ``(g1, g2, offsets, n)``; an empty image (no objects) yields
    ``(None, None, offsets, 0)``. Shared by every colocalization feature so the
    mask-contiguity, ``labels_to_offsets`` and ``flatten_pairs_grouped`` chain
    lives in one place.
    """
    masks = numpy.ascontiguousarray(masks_zyx)
    if not numpy.issubdtype(masks.dtype, numpy.integer):
        masks = masks.astype(numpy.intp)
    lut, n, offsets = labels_to_offsets(masks)
    if n == 0:
        return None, None, offsets, 0
    g1, g2 = flatten_pairs_grouped(
        masks,
        numpy.ascontiguousarray(pixels_1_zyx, dtype=numpy.float64),
        numpy.ascontiguousarray(pixels_2_zyx, dtype=numpy.float64),
        lut,
        offsets,
    )
    return g1, g2, offsets, n


def _run(masks_zyx, pixels_1_zyx, pixels_2_zyx, thr_frac, compute_rwc):
    """Flatten one ``(Z, Y, X)`` image triple and run the fused kernel.

    Returns the nine per-object arrays from ``coloc_per_object``, or ``None`` when
    the image holds no objects (the callers then emit empty feature arrays).
    """
    g1, g2, offsets, n = _flatten_image(masks_zyx, pixels_1_zyx, pixels_2_zyx)
    if n == 0:
        return None
    return coloc_per_object(g1, g2, offsets, n, thr_frac, compute_rwc)


def _featurize(pixels_1, pixels_2, masks, thr, compute_rwc, build):
    """Normalise the triple via ``to_bzyx`` and map each image through ``build``.

    ``build(result)`` turns the kernel tuple (or ``None`` for an empty image) into
    one image's feature dict. ``unwrap`` then collapses a single image to that
    dict, or returns the list for a batch — matching the numba intensity backend.
    """
    frac = thr / 100.0
    masks_list, pixels_1_list, unwrap = to_bzyx(masks, pixels_1)
    _, pixels_2_list, _ = to_bzyx(masks, pixels_2)
    results = [
        build(_run(m, p1, p2, frac, compute_rwc))
        for m, p1, p2 in zip(masks_list, pixels_1_list, pixels_2_list)
    ]
    return unwrap(results)


_EMPTY = numpy.empty(0)


def get_correlation_pearson(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
) -> dict[str, NDArray[numpy.floating]]:
    def build(res):
        if res is None:
            return {F_CORRELATION_FORMAT: _EMPTY, F_SLOPE_FORMAT: _EMPTY}
        corr, slope = res[0], res[1]
        return {F_CORRELATION_FORMAT: corr, F_SLOPE_FORMAT: slope}

    return _featurize(pixels_1, pixels_2, masks, 15, False, build)


def get_correlation_manders_fold(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, NDArray[numpy.floating]]:
    def build(res):
        if res is None:
            return {f"{F_MANDERS_FORMAT}_1": _EMPTY, f"{F_MANDERS_FORMAT}_2": _EMPTY}
        m1, m2 = res[2], res[3]
        return {f"{F_MANDERS_FORMAT}_1": m1, f"{F_MANDERS_FORMAT}_2": m2}

    return _featurize(pixels_1, pixels_2, masks, thr, False, build)


def get_correlation_rwc(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, NDArray[numpy.floating]]:
    def build(res):
        if res is None:
            return {f"{F_RWC_FORMAT}_1": _EMPTY, f"{F_RWC_FORMAT}_2": _EMPTY}
        rwc1, rwc2 = res[7], res[8]
        return {f"{F_RWC_FORMAT}_1": rwc1, f"{F_RWC_FORMAT}_2": rwc2}

    return _featurize(pixels_1, pixels_2, masks, thr, True, build)


def get_correlation_overlap(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, NDArray[numpy.floating]]:
    def build(res):
        if res is None:
            return {
                F_OVERLAP_FORMAT: _EMPTY,
                f"{F_K_FORMAT}_1": _EMPTY,
                f"{F_K_FORMAT}_2": _EMPTY,
            }
        overlap, k1, k2 = res[4], res[5], res[6]
        return {
            F_OVERLAP_FORMAT: overlap,
            f"{F_K_FORMAT}_1": k1,
            f"{F_K_FORMAT}_2": k2,
        }

    return _featurize(pixels_1, pixels_2, masks, thr, False, build)


def get_correlation_costes(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    fast_costes: str = M_FASTER,
    thr: int = 15,
) -> dict[str, NDArray[numpy.floating]]:
    """Costes automated-threshold Manders coefficients C1/C2.

    ``thr`` is accepted for signature parity but has no effect (in the reference it
    only fed the dead ``calculate_threshold`` call). ``scale`` is dtype-derived via
    ``infer_scale`` on ``pixels_1``, so float input gives ``scale == 1``.
    """
    mode = _COSTES_MODE[fast_costes]
    masks_list, pixels_1_list, unwrap = to_bzyx(masks, pixels_1)
    _, pixels_2_list, _ = to_bzyx(masks, pixels_2)

    def run(masks_zyx, p1_zyx, p2_zyx):
        g1, g2, offsets, n = _flatten_image(masks_zyx, p1_zyx, p2_zyx)
        if n == 0:
            return {f"{F_COSTES_FORMAT}_1": _EMPTY, f"{F_COSTES_FORMAT}_2": _EMPTY}
        scale = float(infer_scale(numpy.asarray(p1_zyx)))
        c1, c2 = costes_per_object(g1, g2, offsets, n, scale, mode)
        return {f"{F_COSTES_FORMAT}_1": c1, f"{F_COSTES_FORMAT}_2": c2}

    results = [
        run(m, p1, p2) for m, p1, p2 in zip(masks_list, pixels_1_list, pixels_2_list)
    ]
    return unwrap(results)
