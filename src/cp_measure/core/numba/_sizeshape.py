"""Numba sizeshape kernels (2D).

Phase 1: the spatial-moment matrices (raw + central), which `regionprops_table` computes
per-region with an `einsum` whose contraction path is re-derived for every object. The moments
are plain reductions, so a fused numba pass over the foreground pixels replaces the whole set;
the derived quantities (normalized / Hu / inertia) reuse the shared algebra in
`cp_measure.primitives._moments`. Raw moments are bit-exact vs regionprops; centroid-dependent
matrices match to floating-point round-off.
"""

import numba
import numpy
from numpy.typing import NDArray

from cp_measure.primitives._moments import derive_normalized_hu
from cp_measure.primitives.segment import labels_to_offsets

_ORDER = 4


@numba.njit(cache=True)
def _moment_kernel(
    rows: NDArray[numpy.int64],
    cols: NDArray[numpy.int64],
    obj: NDArray[numpy.int64],
    n: int,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Per-object raw + central spatial moments in two fused passes over foreground pixels.

    Pass A accumulates the 16 raw moments in each object's local (bbox) frame and tracks the
    bbox-min inline; pass B accumulates the 16 central moments in centred coordinates (after the
    per-object centroid is known). Local/centred coordinates and the moment conventions match
    ``skimage`` so the result equals ``regionprops`` to round-off.
    """
    big = 1 << 30
    rmin = numpy.full(n, big, numpy.int64)
    cmin = numpy.full(n, big, numpy.int64)
    for k in range(obj.shape[0]):
        o = obj[k]
        if rows[k] < rmin[o]:
            rmin[o] = rows[k]
        if cols[k] < cmin[o]:
            cmin[o] = cols[k]

    raw = numpy.zeros((n, _ORDER, _ORDER))
    for k in range(obj.shape[0]):
        o = obj[k]
        lr = float(rows[k] - rmin[o])
        lc = float(cols[k] - cmin[o])
        rp = 1.0
        for p in range(_ORDER):
            cp = 1.0
            for q in range(_ORDER):
                raw[o, p, q] += rp * cp
                cp *= lc
            rp *= lr

    centre_r = numpy.empty(n)
    centre_c = numpy.empty(n)
    for o in range(n):
        centre_r[o] = raw[o, 1, 0] / raw[o, 0, 0]
        centre_c[o] = raw[o, 0, 1] / raw[o, 0, 0]

    central = numpy.zeros((n, _ORDER, _ORDER))
    for k in range(obj.shape[0]):
        o = obj[k]
        dr = (rows[k] - rmin[o]) - centre_r[o]
        dc = (cols[k] - cmin[o]) - centre_c[o]
        rp = 1.0
        for p in range(_ORDER):
            cp = 1.0
            for q in range(_ORDER):
                central[o, p, q] += rp * cp
                cp *= dc
            rp *= dr

    return raw, central


def spatial_moments_2d(
    labels: NDArray[numpy.integer],
) -> tuple[
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
]:
    """numba accumulator: per-object ``(raw, central, normalized, hu)`` moments (2D).

    Drop-in for ``cp_measure.primitives._moments.spatial_moments_2d`` (same object order, same
    derivation), but computes the moment matrices with the fused numba kernel instead of 32
    ``numpy.bincount`` passes.
    """
    lut, n, _offsets = labels_to_offsets(labels)
    if n == 0:
        empty = numpy.zeros((0, _ORDER, _ORDER))
        return empty, empty, empty, numpy.zeros((0, 7))
    rows, cols = numpy.nonzero(labels)
    obj = lut[labels[rows, cols]].astype(numpy.int64)
    raw, central = _moment_kernel(
        rows.astype(numpy.int64), cols.astype(numpy.int64), obj, n
    )
    normalized, hu = derive_normalized_hu(central)
    return raw, central, normalized, hu
