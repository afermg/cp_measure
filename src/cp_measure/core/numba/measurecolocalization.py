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

from collections.abc import Iterable

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


_EMPTY = numpy.empty(0)


# The five single-feature functions are thin, gated wrappers over ``get_correlation_all`` (single
# source of truth for the kernel + feature-key assembly). Each computes only its tier: the cheap
# functions skip RWC's rank sort and the Costes kernel; ``rwc`` adds the rank sort; ``costes`` runs
# only the iterative kernel. Calling several separately still runs the kernel per call (the library
# is stateless) — pass the set to ``get_correlation_all`` to collapse them into one pass.


def get_correlation_pearson(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
) -> dict[str, NDArray[numpy.floating]]:
    return get_correlation_all(pixels_1, pixels_2, masks, features=("pearson",))


def get_correlation_manders_fold(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, NDArray[numpy.floating]]:
    return get_correlation_all(
        pixels_1, pixels_2, masks, features=("manders_fold",), thr=thr
    )


def get_correlation_rwc(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, NDArray[numpy.floating]]:
    return get_correlation_all(pixels_1, pixels_2, masks, features=("rwc",), thr=thr)


def get_correlation_overlap(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, NDArray[numpy.floating]]:
    return get_correlation_all(
        pixels_1, pixels_2, masks, features=("overlap",), thr=thr
    )


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
    return get_correlation_all(
        pixels_1,
        pixels_2,
        masks,
        features=("costes",),
        thr=thr,
        fast_costes=fast_costes,
    )


_GROUPS = ("pearson", "manders_fold", "rwc", "overlap", "costes")
_CHEAP_GROUPS = frozenset({"pearson", "manders_fold", "rwc", "overlap"})


def get_correlation_all(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    features: Iterable[str] | None = None,
    thr: int = 15,
    fast_costes: str = M_FASTER,
) -> dict[str, NDArray[numpy.floating]]:
    """Colocalization features from ONE flatten + ONE fused kernel pass per image.

    ``features`` selects which groups to return — any of ``pearson`` / ``manders_fold`` / ``rwc`` /
    ``overlap`` / ``costes`` — or ``None`` for all. The shared ``coloc_per_object`` kernel runs once
    for the cheap block (Pearson + slope, Manders, Overlap, K); the two expensive paths are gated by
    the request: RWC's rank sort runs only if ``rwc`` is asked for, the Costes iterative kernel only
    if ``costes`` is. This is the efficient entry point for any caller wanting several coloc features
    at once (the single-feature functions delegate here). Stateless: collapsing N features into one
    pass means requesting them in one call.
    """
    want = set(_GROUPS) if features is None else set(features)
    unknown = want - set(_GROUPS)
    if unknown:
        raise ValueError(f"unknown correlation feature group(s): {sorted(unknown)}")
    need_cheap = bool(want & _CHEAP_GROUPS)
    need_rwc = "rwc" in want
    need_costes = "costes" in want
    frac = thr / 100.0
    mode = _COSTES_MODE[fast_costes]
    masks_list, pixels_1_list, unwrap = to_bzyx(masks, pixels_1)
    _, pixels_2_list, _ = to_bzyx(masks, pixels_2)

    def run(masks_zyx, p1_zyx, p2_zyx):
        g1, g2, offsets, n = _flatten_image(masks_zyx, p1_zyx, p2_zyx)
        out: dict[str, NDArray[numpy.floating]] = {}
        if need_cheap or need_rwc:
            if n == 0:
                corr = slope = m1 = m2 = overlap = k1 = k2 = rwc1 = rwc2 = _EMPTY
            else:
                corr, slope, m1, m2, overlap, k1, k2, rwc1, rwc2 = coloc_per_object(
                    g1, g2, offsets, n, frac, need_rwc
                )
            if "pearson" in want:
                out[F_CORRELATION_FORMAT] = corr
                out[F_SLOPE_FORMAT] = slope
            if "manders_fold" in want:
                out[f"{F_MANDERS_FORMAT}_1"] = m1
                out[f"{F_MANDERS_FORMAT}_2"] = m2
            if "overlap" in want:
                out[F_OVERLAP_FORMAT] = overlap
                out[f"{F_K_FORMAT}_1"] = k1
                out[f"{F_K_FORMAT}_2"] = k2
            if need_rwc:
                out[f"{F_RWC_FORMAT}_1"] = rwc1
                out[f"{F_RWC_FORMAT}_2"] = rwc2
        if need_costes:
            if n == 0:
                c1 = c2 = _EMPTY
            else:
                scale = float(infer_scale(numpy.asarray(p1_zyx)))
                c1, c2 = costes_per_object(g1, g2, offsets, n, scale, mode)
            out[f"{F_COSTES_FORMAT}_1"] = c1
            out[f"{F_COSTES_FORMAT}_2"] = c2
        return out

    results = [
        run(m, p1, p2) for m, p1, p2 in zip(masks_list, pixels_1_list, pixels_2_list)
    ]
    return unwrap(results)
