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
from skimage.measure import grid_points_in_poly

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


# --- Convex hull (area_convex / solidity) --------------------------------------------------
# skimage's `area_convex` is the pixel count inside the convex hull of each object's pixels,
# where each pixel contributes its 4 edge-midpoints (±0.5 "diamond" offsets) and the count comes
# from `grid_points_in_poly`. We replace only the slow per-region hull *construction* (scipy
# QHull) with a fused numba monotone-chain over each object's BOUNDARY pixels (hull(boundary) ==
# hull(object); ~6-17x fewer points), and keep skimage's exact `grid_points_in_poly` raster — so
# the result is bit-exact (proven 142/142). Coordinates are scaled x2 so the offset points are
# integers and the hull is exact.


def _boundary_mask(masks: NDArray[numpy.integer]) -> NDArray[numpy.bool_]:
    """Foreground pixels with an 8-neighbour of a different label (or the image edge).

    The convex hull of these equals the hull of the whole object (interior pixels are never hull
    vertices), so feeding only boundary pixels shrinks the hull input dramatically.
    """
    height, width = masks.shape
    padded = numpy.pad(masks, 1)
    foreground = masks > 0
    all_same = numpy.ones_like(foreground)
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_row == 0 and d_col == 0:
                continue
            shifted = padded[
                1 + d_row : height + 1 + d_row, 1 + d_col : width + 1 + d_col
            ]
            all_same &= shifted == masks
    return foreground & ~all_same


@numba.njit(cache=True)
def _hull_kernel(
    px: NDArray[numpy.int64],
    py: NDArray[numpy.int64],
    offsets: NDArray[numpy.int64],
    n: int,
    stride: int,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.int64]]:
    """Per-object convex hull (Andrew's monotone chain) over grouped integer points.

    Points are the x2-scaled diamond-offset boundary points of each object, grouped by object via
    ``offsets`` (CSR). Returns the hull vertices (divided back by 2) flat with per-object offsets.
    """
    total = px.shape[0]
    out_x = numpy.empty(total, numpy.float64)
    out_y = numpy.empty(total, numpy.float64)
    hull_offsets = numpy.zeros(n + 1, numpy.int64)
    cur = 0
    for o in range(n):
        start = offsets[o]
        end = offsets[o + 1]
        m = end - start
        if m == 0:
            hull_offsets[o + 1] = cur
            continue
        key = px[start:end] * stride + py[start:end]
        order = numpy.argsort(key)
        sx = px[start:end][order]
        sy = py[start:end][order]
        # dedup consecutive identical points (sorted -> duplicates are adjacent)
        ux = numpy.empty(m, numpy.int64)
        uy = numpy.empty(m, numpy.int64)
        u = 0
        for i in range(m):
            if u == 0 or sx[i] != ux[u - 1] or sy[i] != uy[u - 1]:
                ux[u] = sx[i]
                uy[u] = sy[i]
                u += 1
        if u < 3:
            for i in range(u):
                out_x[cur] = ux[i] / 2.0
                out_y[cur] = uy[i] / 2.0
                cur += 1
            hull_offsets[o + 1] = cur
            continue
        hx = numpy.empty(2 * u, numpy.int64)
        hy = numpy.empty(2 * u, numpy.int64)
        k = 0
        for i in range(u):  # lower hull
            while (
                k >= 2
                and (hx[k - 1] - hx[k - 2]) * (uy[i] - hy[k - 2])
                - (hy[k - 1] - hy[k - 2]) * (ux[i] - hx[k - 2])
                <= 0
            ):
                k -= 1
            hx[k] = ux[i]
            hy[k] = uy[i]
            k += 1
        lower_end = k + 1
        for i in range(u - 2, -1, -1):  # upper hull
            while (
                k >= lower_end
                and (hx[k - 1] - hx[k - 2]) * (uy[i] - hy[k - 2])
                - (hy[k - 1] - hy[k - 2]) * (ux[i] - hx[k - 2])
                <= 0
            ):
                k -= 1
            hx[k] = ux[i]
            hy[k] = uy[i]
            k += 1
        for i in range(k - 1):  # drop the repeated closing vertex
            out_x[cur] = hx[i] / 2.0
            out_y[cur] = hy[i] / 2.0
            cur += 1
        hull_offsets[o + 1] = cur
    return out_x[:cur], out_y[:cur], hull_offsets


def convex_area_2d(labels: NDArray[numpy.integer]) -> NDArray[numpy.floating]:
    """Per-object ``area_convex`` (pixel count inside the convex hull), ordered by ascending
    label. Bit-exact vs ``skimage.measure.regionprops``."""
    lut, n, offsets = labels_to_offsets(labels)
    if n == 0:
        return numpy.zeros(0)
    bnd = _boundary_mask(labels)
    rows, cols = numpy.nonzero(bnd)
    obj = lut[labels[rows, cols]]
    # per-object bbox (extremes are boundary pixels, so this equals the full-object bbox)
    rmin = numpy.full(n, 1 << 30)
    cmin = numpy.full(n, 1 << 30)
    rmax = numpy.full(n, -1)
    cmax = numpy.full(n, -1)
    numpy.minimum.at(rmin, obj, rows)
    numpy.minimum.at(cmin, obj, cols)
    numpy.maximum.at(rmax, obj, rows)
    numpy.maximum.at(cmax, obj, cols)
    # 4 diamond offsets per boundary pixel, x2-scaled to integers, grouped by object
    r2 = rows.astype(numpy.int64) * 2
    c2 = cols.astype(numpy.int64) * 2
    px = numpy.concatenate([r2 - 1, r2 + 1, r2, r2])
    py = numpy.concatenate([c2, c2, c2 - 1, c2 + 1])
    obj4 = numpy.concatenate([obj, obj, obj, obj])
    order = numpy.argsort(obj4, kind="stable")
    px, py, obj4 = px[order], py[order], obj4[order]
    grp = numpy.zeros(n + 1, numpy.int64)
    numpy.add.at(grp, obj4 + 1, 1)
    grp = numpy.cumsum(grp)
    stride = numpy.int64(4 * max(labels.shape) + 10)
    hx, hy, hoff = _hull_kernel(px, py, grp, n, stride)

    counts = numpy.diff(offsets)  # full per-object pixel counts (for degenerate hulls)
    area = numpy.empty(n)
    for o in range(n):
        verts = numpy.column_stack(
            [hx[hoff[o] : hoff[o + 1]] - rmin[o], hy[hoff[o] : hoff[o + 1]] - cmin[o]]
        )
        if len(verts) < 3:
            area[o] = counts[o]  # point / line: hull is the pixels themselves
        else:
            shape = (int(rmax[o] - rmin[o] + 1), int(cmax[o] - cmin[o] + 1))
            area[o] = (grid_points_in_poly(shape, verts, binarize=False) >= 1).sum()
    return area
