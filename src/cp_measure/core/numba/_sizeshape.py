"""Numba 2D size/shape geometry, isolated from PR #78's integration-stack base.

Accelerates convex hull construction and rasterization, label-aware perimeter/topology
reductions, and a per-object bbox pass (bbox, centroid, filled area, Euclidean distance
radii) that replaces regionprops and scipy's distance transform with bit-identical results.
Spatial moments are summed in the NumPy backend's separable order (per row, then over
rows) without the GIL; they match it to round-off, as BLAS may reorder the NumPy sums.
"""

from typing import NamedTuple, cast

import numba
import numpy
from numpy.typing import NDArray

from cp_measure.core import measureobjectsizeshape as _ss
from cp_measure.primitives.parallel import map_images
from cp_measure.primitives.shapes import is_batch, per_image
from cp_measure.primitives._segment_numba import numpy_sum, numpy_sum_chunk
from cp_measure.primitives._moments import (
    axes_eccentricity_orientation,
    inertia_2d,
    moment_feature_dict,
    normalized_and_hu,
)


class _Prep(NamedTuple):
    """Label lookup and foreground counts shared by the geometry kernels."""

    lut: NDArray[numpy.int64]
    n: int
    offsets: NDArray[numpy.int64]


def _as_int64(labels: NDArray[numpy.integer]) -> NDArray[numpy.int64]:
    """Native-endian, C-contiguous int64 labels: the one signature the kernels compile for.
    No copy when ``labels`` already is."""
    return numpy.ascontiguousarray(labels, dtype=numpy.int64)


def _foreground_prep(labels: NDArray[numpy.integer]) -> _Prep:
    """One counting pass; geometry primitives also accept sparse positive labels."""
    lut, n, offsets = _prep_kernel(_as_int64(labels).ravel())
    return _Prep(lut, n, offsets)


@numba.njit(cache=True, nogil=True)
def _prep_kernel(flat):
    """Count the foreground labels, then build a label -> object lookup (-1 if absent) and
    the CSR offsets of each present label's pixel count, in ascending label order. Rejects
    negative labels with ``numpy.bincount``'s error, as before."""
    top = -1
    for v in flat:
        if v < 0:
            raise ValueError("'list' argument must have no negative elements")
        top = max(top, v)
    counts = numpy.zeros(top + 1, numpy.int64)
    for v in flat:
        if v > 0:  # background is unused; counting it serializes on one counter
            counts[v] += 1
    lut = numpy.full(top + 1, -1, numpy.int64)
    n = 0
    for lab in range(1, top + 1):
        if counts[lab] > 0:
            lut[lab] = n
            n += 1
    offsets = numpy.zeros(n + 1, numpy.int64)
    for lab in range(1, top + 1):
        if counts[lab] > 0:
            offsets[lut[lab] + 1] = offsets[lut[lab]] + counts[lab]
    return lut, n, offsets


# --- Convex hull (area_convex / solidity) --------------------------------------------------
# skimage's `area_convex` is the pixel count inside the convex hull of each object's pixels,
# where each pixel contributes its 4 edge-midpoints (±0.5 "diamond" offsets) and the count comes
# from `grid_points_in_poly`. We replace the per-region hull construction (scipy QHull) with a
# fused numba monotone-chain over each object's BOUNDARY pixels (hull(boundary) ==
# hull(object); ~6-17x fewer points), and the raster with an exact integer scanline count that
# keeps skimage's pixel-inclusion convention (inside, vertex or edge). Coordinates are scaled x2
# so offsets are integers and the hull is exact.


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


@numba.njit(cache=True, nogil=True)
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


def convex_area_2d(
    labels: NDArray[numpy.integer], prep: _Prep | None = None
) -> NDArray[numpy.floating]:
    """Per-object ``area_convex`` (pixel count inside the convex hull), ordered by ascending
    label. Matches ``skimage.measure.regionprops`` pixel counts. Reuses ``prep``'s ``lut``/``n``/
    ``offsets`` when supplied; the boundary scan below is its own (smaller) pass."""
    if prep is None:
        prep = _foreground_prep(labels)
    lut, n, offsets = prep.lut, prep.n, prep.offsets
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
    order, grp = _group_by_object(obj4, n)
    px, py = px[order], py[order]
    stride = 4 * max(labels.shape) + 10
    hx, hy, hoff = _hull_kernel(px, py, grp, n, stride)
    counts = numpy.diff(offsets)  # full per-object pixel counts (for degenerate hulls)
    return _hull_areas(hx, hy, hoff, rmin, cmin, rmax, cmax, counts)


@numba.njit(cache=True, nogil=True)
def _group_by_object(obj, n):
    """Stable counting sort by object: the order of ``argsort(obj, kind="stable")``,
    plus the CSR offsets of each object's group."""
    grp = numpy.zeros(n + 1, numpy.int64)
    for o in obj:
        grp[o + 1] += 1
    for o in range(n):
        grp[o + 1] += grp[o]
    cursor = grp[:-1].copy()
    order = numpy.empty(obj.size, numpy.int64)
    for i in range(obj.size):
        order[cursor[obj[i]]] = i
        cursor[obj[i]] += 1
    return order, grp


@numba.njit(cache=True, nogil=True)
def _hull_areas(hx, hy, hoff, rmin, cmin, rmax, cmax, counts):
    """Per-object grid-point count of the closed hull polygon, i.e. skimage's
    ``grid_points_in_poly(..., binarize=False) >= 1`` (inside, vertex or edge).

    Hull vertices lie on the half-pixel grid, so doubling them gives integers and each
    scanline's crossings are exact rationals: per row, the closed polygon spans
    ``[min ceil(crossing), max floor(crossing)]`` over the edges that reach the row.
    """
    n = hoff.size - 1
    area = numpy.empty(n)
    for o in range(n):
        start, end = hoff[o], hoff[o + 1]
        nv = end - start
        if nv < 3:
            area[o] = counts[o]  # point / line: hull is the pixels themselves
            continue
        height = rmax[o] - rmin[o] + 1
        width = cmax[o] - cmin[o] + 1
        total = 0
        for r in range(height):
            y = 2 * r
            lo = width
            hi = -1
            for e in range(nv):
                f = start + (e + 1) % nv
                r1 = int(2 * (hx[start + e] - rmin[o]))
                c1 = int(2 * (hy[start + e] - cmin[o]))
                r2 = int(2 * (hx[f] - rmin[o]))
                c2 = int(2 * (hy[f] - cmin[o]))
                if (y < r1 and y < r2) or (y > r1 and y > r2):
                    continue
                if r1 == r2:  # horizontal edge on this row: both endpoints
                    a, b = min(c1, c2), max(c1, c2)
                    col_lo, col_hi = -((-a) // 2), b // 2
                else:  # doubled crossing column num / den, den > 0
                    den = r2 - r1
                    num = c1 * den + (y - r1) * (c2 - c1)
                    if den < 0:
                        den, num = -den, -num
                    col_lo, col_hi = -((-num) // (2 * den)), num // (2 * den)
                lo = min(lo, col_lo)
                hi = max(hi, col_hi)
            lo = max(lo, 0)
            hi = min(hi, width - 1)
            if hi >= lo:
                total += hi - lo + 1
        area[o] = total
    return area


# --- Perimeter, perimeter_crofton, euler_number ---------------------------------------------
# All three are deterministic neighbour-pattern reductions skimage runs per-region with C
# convolutions. A whole-image label-aware numba pass reproduces their pattern counts: each object's
# pattern histogram is identical to running skimage on the isolated object (other labels read as
# background), and skimage's per-region 1-pixel pad is the whole-image edge pad.

# perimeter (4-connectivity): skimage convolves the border image with [[10,2,10],[2,1,2],[10,2,10]]
# and weights the histogram. Only border-centre pixels (odd values) carry nonzero weight; a border
# pixel's value is 1 + 2*(same-object 4-conn border) + 10*(same-object diagonal border).
_PERIMETER_WEIGHTS = numpy.zeros(50)
_PERIMETER_WEIGHTS[[5, 7, 15, 17, 25, 27]] = 1.0
_PERIMETER_WEIGHTS[[21, 33]] = numpy.sqrt(2)
_PERIMETER_WEIGHTS[[13, 23]] = (1 + numpy.sqrt(2)) / 2

# crofton (4 directions) and euler (8-connectivity) share the 2x2 binary-config histogram
# (skimage's XF kernel [[0,0,0],[0,1,4],[0,2,8]], 16 bins); only the coefficients differ.
_CROFTON_COEFS_4 = numpy.array(
    [
        0.0,
        numpy.pi / 4 * (1 + 1 / numpy.sqrt(2)),
        numpy.pi / (4 * numpy.sqrt(2)),
        numpy.pi / (2 * numpy.sqrt(2)),
        0.0,
        numpy.pi / 4 * (1 + 1 / numpy.sqrt(2)),
        0.0,
        numpy.pi / (4 * numpy.sqrt(2)),
        numpy.pi / 4,
        numpy.pi / 2,
        numpy.pi / (4 * numpy.sqrt(2)),
        numpy.pi / (4 * numpy.sqrt(2)),
        numpy.pi / 4,
        numpy.pi / 2,
        0.0,
        0.0,
    ]
)
_EULER_COEFS_8 = numpy.array(
    [0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 0], dtype=numpy.int64
)


@numba.njit(cache=True, nogil=True)
def _perimeter_kernel(
    labels: NDArray[numpy.integer],
    lut: NDArray[numpy.int64],
    n: int,
    weights: NDArray[numpy.float64],
) -> NDArray[numpy.float64]:
    """Per-object 4-connectivity perimeter (skimage's border-pattern weighting)."""
    height, width = labels.shape
    border = numpy.zeros((height, width), numpy.uint8)
    for r in range(height):
        for c in range(width):
            lab = labels[r, c]
            if lab <= 0:
                continue
            same = (
                (r > 0 and labels[r - 1, c] == lab)
                and (r < height - 1 and labels[r + 1, c] == lab)
                and (c > 0 and labels[r, c - 1] == lab)
                and (c < width - 1 and labels[r, c + 1] == lab)
            )
            if not same:
                border[r, c] = 1

    out = numpy.zeros(n)
    for r in range(height):
        for c in range(width):
            if border[r, c] == 0:
                continue
            lab = labels[r, c]
            edges = 0
            corners = 0
            if r > 0 and border[r - 1, c] and labels[r - 1, c] == lab:
                edges += 1
            if r < height - 1 and border[r + 1, c] and labels[r + 1, c] == lab:
                edges += 1
            if c > 0 and border[r, c - 1] and labels[r, c - 1] == lab:
                edges += 1
            if c < width - 1 and border[r, c + 1] and labels[r, c + 1] == lab:
                edges += 1
            if r > 0 and c > 0 and border[r - 1, c - 1] and labels[r - 1, c - 1] == lab:
                corners += 1
            if (
                r > 0
                and c < width - 1
                and border[r - 1, c + 1]
                and labels[r - 1, c + 1] == lab
            ):
                corners += 1
            if (
                r < height - 1
                and c > 0
                and border[r + 1, c - 1]
                and labels[r + 1, c - 1] == lab
            ):
                corners += 1
            if (
                r < height - 1
                and c < width - 1
                and border[r + 1, c + 1]
                and labels[r + 1, c + 1] == lab
            ):
                corners += 1
            out[lut[lab]] += weights[1 + 2 * edges + 10 * corners]
    return out


@numba.njit(cache=True, nogil=True)
def _xf_hist_kernel(
    padded: NDArray[numpy.integer], lut: NDArray[numpy.int64], n: int
) -> NDArray[numpy.int64]:
    """Per-object 2x2 binary-config histogram (16 bins), label-aware over the padded image.

    For each 2x2 window and each distinct positive label in it, the config bit pattern (matching
    skimage's ``XF`` convolution) is incremented for that label — equal to running skimage's
    convolution on each isolated, 1-pixel-padded object.
    """
    h_pad, w_pad = padded.shape
    hist = numpy.zeros((n, 16), numpy.int64)
    for i in range(h_pad):
        for j in range(w_pad):
            a = padded[i, j]
            b = padded[i, j - 1] if j > 0 else 0
            c = padded[i - 1, j] if i > 0 else 0
            d = padded[i - 1, j - 1] if (i > 0 and j > 0) else 0
            # each distinct positive label in the 2x2 window contributes its config once
            if a > 0:
                hist[
                    lut[a],
                    1
                    + (4 if b == a else 0)
                    + (2 if c == a else 0)
                    + (8 if d == a else 0),
                ] += 1
            if b > 0 and b != a:
                hist[
                    lut[b],
                    (1 if a == b else 0)
                    + 4
                    + (2 if c == b else 0)
                    + (8 if d == b else 0),
                ] += 1
            if c > 0 and c != a and c != b:
                hist[
                    lut[c],
                    (1 if a == c else 0)
                    + (4 if b == c else 0)
                    + 2
                    + (8 if d == c else 0),
                ] += 1
            if d > 0 and d != a and d != b and d != c:
                hist[
                    lut[d],
                    (1 if a == d else 0)
                    + (4 if b == d else 0)
                    + (2 if c == d else 0)
                    + 8,
                ] += 1
    return hist


def perimeter_2d(
    labels: NDArray[numpy.integer], prep: _Prep | None = None
) -> NDArray[numpy.floating]:
    """Per-object 4-connectivity perimeter, matching skimage to round-off."""
    if prep is None:
        prep = _foreground_prep(labels)
    lut, n = prep.lut, prep.n
    if n == 0:
        return numpy.zeros(0)
    # Numba requires native byte order; signed labels also avoid uint64/int64
    # promotion to float in the neighbor kernel's background sentinel branches.
    return _perimeter_kernel(_as_int64(labels), lut, n, _PERIMETER_WEIGHTS)


def crofton_euler_2d(
    labels: NDArray[numpy.integer], prep: _Prep | None = None
) -> tuple[NDArray[numpy.floating], NDArray[numpy.int64]]:
    """Per-object ``(perimeter_crofton, euler_number)`` from the shared 2x2-config histogram."""
    if prep is None:
        prep = _foreground_prep(labels)
    lut, n = prep.lut, prep.n
    if n == 0:
        return numpy.zeros(0), numpy.zeros(0, dtype=numpy.int64)
    padded = numpy.pad(_as_int64(labels), 1)
    hist = _xf_hist_kernel(padded, lut, n)
    return hist @ _CROFTON_COEFS_4, hist @ _EULER_COEFS_8


# --- Per-object bbox pass: bbox, centroid, filled area, EDT radii --------------------------
# Replaces regionprops_table's area / bbox / centroid / area_filled and the per-object scipy
# EDT radius loop with one pass over each object's bounding box. Every output is bit-identical:
# EDT distances are sqrt of exact integer squared distances (as scipy's), the centroid is the
# serial coordinate sum numpy uses for an axis-0 mean, and holes are filled with skimage's full
# 3x3 structure, and the mean radius is NumPy's pairwise sum (``numpy_sum``) over the count.


@numba.njit(cache=True, nogil=True)
def _squared_edt_row(g, mask_row, width, v, t, out_row):
    """Meijster's second phase on one row: squared distance to the nearest background pixel,
    given ``g`` (per-column vertical distances), written to ``out_row`` where ``mask_row``."""
    q = 0
    v[0] = 0
    t[0] = 0
    for u in range(1, width):
        gu = g[u] * g[u]
        while q >= 0:
            s = v[q]
            if (t[q] - s) * (t[q] - s) + g[s] * g[s] <= (t[q] - u) * (t[q] - u) + gu:
                break
            q -= 1
        if q < 0:
            q = 0
            v[0] = u
        else:
            s = v[q]
            sep = (u * u - s * s + gu - g[s] * g[s]) // (2 * (u - s)) + 1
            if sep < width:
                q += 1
                v[q] = u
                t[q] = sep
    for u in range(width - 1, -1, -1):
        s = v[q]
        if mask_row[u]:
            out_row[u] = (u - s) * (u - s) + g[s] * g[s]
        if u == t[q]:
            q -= 1


@numba.njit(cache=True, nogil=True)
def _object_geometry(labels, lut, n, chunk):
    """Per object (ascending label): bbox, pixel count, centroid, filled area, and the max,
    mean and median EDT radius.

    The EDT runs on the object's bbox padded by one background pixel, as the numpy backend's
    ``distance_transform_edt(numpy.pad(image, 1))``; filled area floods the unpadded bbox
    background from its border with 8-connectivity (skimage's ``image_filled``).
    """
    height, width = labels.shape
    r0 = numpy.full(n, height, numpy.int64)
    c0 = numpy.full(n, width, numpy.int64)
    r1 = numpy.zeros(n, numpy.int64)
    c1 = numpy.zeros(n, numpy.int64)
    count = numpy.zeros(n, numpy.int64)
    for r in range(height):
        for c in range(width):
            if labels[r, c] > 0:
                o = lut[labels[r, c]]
                count[o] += 1
                r0[o] = min(r0[o], r)
                c0[o] = min(c0[o], c)
                r1[o] = max(r1[o], r + 1)
                c1[o] = max(c1[o], c + 1)
    offsets = numpy.zeros(n + 1, numpy.int64)
    max_h = max_w = 1
    for o in range(n):
        offsets[o + 1] = offsets[o] + count[o]
        max_h = max(max_h, r1[o] - r0[o] + 2)
        max_w = max(max_w, c1[o] - c0[o] + 2)
    dist = numpy.empty(offsets[n])
    filled = numpy.zeros(n, numpy.int64)
    max_radius = numpy.zeros(n)
    median_radius = numpy.zeros(n)
    mean_radius = numpy.zeros(n)
    center_y = numpy.zeros(n)
    center_x = numpy.zeros(n)
    mask_buf = numpy.zeros((max_h, max_w), numpy.bool_)
    g_buf = numpy.empty((max_h, max_w), numpy.int64)
    d2_row = numpy.empty(max_w, numpy.int64)
    seen_buf = numpy.zeros((max_h, max_w), numpy.bool_)
    stack = numpy.empty(max_h * max_w, numpy.int64)
    v = numpy.empty(max_w, numpy.int64)
    t = numpy.empty(max_w, numpy.int64)
    for o in range(n):
        h = r1[o] - r0[o] + 2  # padded by one pixel on each side
        w = c1[o] - c0[o] + 2
        mask = mask_buf[:h, :w]
        mask[:, :] = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                lab = labels[r0[o] + i - 1, c0[o] + j - 1]
                mask[i, j] = lab > 0 and lut[lab] == o
        # Meijster phase 1: vertical distance to background per column (padding bounds it)
        g = g_buf[:h, :w]
        for j in range(w):
            g[0, j] = 0
            for i in range(1, h):
                g[i, j] = g[i - 1, j] + 1 if mask[i, j] else 0
            for i in range(h - 2, -1, -1):
                if g[i + 1, j] < g[i, j]:
                    g[i, j] = g[i + 1, j] + 1
        k = offsets[o]
        sum_y = 0.0
        sum_x = 0.0
        for i in range(h):
            _squared_edt_row(g[i], mask[i], w, v, t, d2_row)
            for j in range(w):
                if mask[i, j]:
                    dist[k] = numpy.sqrt(numpy.float64(d2_row[j]))
                    sum_y += r0[o] + i - 1
                    sum_x += c0[o] + j - 1
                    k += 1
        center_y[o] = sum_y / count[o]
        center_x[o] = sum_x / count[o]
        inside = dist[offsets[o] : offsets[o + 1]]
        max_radius[o] = inside.max()
        # the numpy backend's ``inside.mean()``: NumPy's pairwise sum over the count
        mean_radius[o] = numpy_sum(inside, chunk) / count[o]
        ordered = numpy.sort(inside)
        mid = count[o] // 2
        median_radius[o] = (
            ordered[mid] if count[o] % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        )
        # holes: background of the unpadded bbox not 8-connected to its border
        bh, bw = h - 2, w - 2
        seen = seen_buf[:bh, :bw]
        seen[:, :] = False
        top = 0
        for i in range(bh):
            for j in range(bw):
                border = i == 0 or j == 0 or i == bh - 1 or j == bw - 1
                if border and not mask[i + 1, j + 1]:
                    seen[i, j] = True
                    stack[top] = i * bw + j
                    top += 1
        reached = top
        while top > 0:
            top -= 1
            i, j = stack[top] // bw, stack[top] % bw
            for a in range(max(i - 1, 0), min(i + 2, bh)):
                for b in range(max(j - 1, 0), min(j + 2, bw)):
                    if not seen[a, b] and not mask[a + 1, b + 1]:
                        seen[a, b] = True
                        stack[top] = a * bw + b
                        top += 1
                        reached += 1
        filled[o] = bh * bw - reached
    bbox = (r0, c0, r1, c1)
    return (
        bbox,
        count,
        center_y,
        center_x,
        filled,
        max_radius,
        mean_radius,
        median_radius,
    )


# --- Spatial moments ---------------------------------------------------------------------------
# The NumPy backend's `spatial_moments_2d` stacks every object's bbox mask in Python and sums
# with BLAS; these kernels take the same per-object sums straight from the label image without
# the GIL. Each sum runs as the NumPy matmuls do: x powers summed along a row, then weighted by
# that row's y powers and summed over rows.


@numba.njit(cache=True, nogil=True)
def _moment_sums(labels, lut, n, r0, c0, shift_r, shift_c):
    """Per object, ``sum y**p * x**q`` for ``p, q < 4`` with ``y = (r - r0) - shift_r`` and
    ``x = (c - c0) - shift_c``: one raster pass, flushing an object's row sums when it moves to a
    new row."""
    height, width = labels.shape
    sums = numpy.zeros((n, 4, 4))
    row_sums = numpy.zeros((n, 4))
    last_row = numpy.full(n, -1, numpy.int64)
    for r in range(height):
        for c in range(width):
            lab = labels[r, c]
            if lab <= 0:
                continue
            o = lut[lab]
            if last_row[o] != r:
                if last_row[o] >= 0:
                    _flush_row(sums, row_sums, o, (last_row[o] - r0[o]) - shift_r[o])
                last_row[o] = r
            x = (c - c0[o]) - shift_c[o]
            x2 = x * x
            row_sums[o, 0] += 1.0
            row_sums[o, 1] += x
            row_sums[o, 2] += x2
            row_sums[o, 3] += x2 * x
    for o in range(n):  # the last row of every object
        if last_row[o] >= 0:
            _flush_row(sums, row_sums, o, (last_row[o] - r0[o]) - shift_r[o])
    return sums


@numba.njit(cache=True, nogil=True)
def _flush_row(sums, row_sums, o, y):
    """Add one row's x-power sums, weighted by ``y**p``, to object ``o`` and reset them."""
    y2 = y * y
    y_pow = (1.0, y, y2, y2 * y)
    for p in range(4):
        for q in range(4):
            sums[o, p, q] += y_pow[p] * row_sums[o, q]
    row_sums[o, :] = 0.0


@numba.njit(cache=True, nogil=True)
def _raw_central_moments(labels, lut, n):
    """Per-object ``(raw, central)`` moments ``(n, 4, 4)`` in each object's bbox frame."""
    height, width = labels.shape
    r0 = numpy.full(n, height, numpy.int64)
    c0 = numpy.full(n, width, numpy.int64)
    for r in range(height):
        for c in range(width):
            if labels[r, c] > 0:
                o = lut[labels[r, c]]
                r0[o] = min(r0[o], r)
                c0[o] = min(c0[o], c)
    zeros = numpy.zeros(n)
    raw = _moment_sums(labels, lut, n, r0, c0, zeros, zeros)
    centre_r = raw[:, 1, 0] / raw[:, 0, 0]
    centre_c = raw[:, 0, 1] / raw[:, 0, 0]
    # Central moments sum the centred coordinates directly: deriving them from the raw
    # moments loses precision to cancellation.
    central = _moment_sums(labels, lut, n, r0, c0, centre_r, centre_c)
    return raw, central


def spatial_moments_2d(
    labels: NDArray[numpy.integer], prep: _Prep | None = None
) -> tuple[
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
]:
    """Per-object ``(raw, central, normalized, hu)`` spatial moments, as the NumPy backend's
    :func:`cp_measure.primitives._moments.spatial_moments_2d`, ordered by ascending label."""
    if prep is None:
        prep = _foreground_prep(labels)
    if prep.n == 0:
        empty = numpy.zeros((0, 4, 4))
        return empty, empty, empty, numpy.zeros((0, 7))
    raw, central = _raw_central_moments(_as_int64(labels), prep.lut, prep.n)
    return raw, central, *normalized_and_hu(central)


# --- Full get_sizeshape wrapper -------------------------------------------------------------
# Assembles the complete sizeshape feature dict from the numba kernels above. With option B
# (axes/eccentricity/orientation derived from the central moments), no moments go through
# regionprops.


def _sizeshape_2d(
    labels: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating] | None,
    calculate_advanced: bool,
    new_features: bool,
) -> dict[str, NDArray]:
    # Convert once: every helper below then gets the labels without a copy.
    labels = _as_int64(labels)
    prep = _foreground_prep(labels)
    # The pixels are unused, but the NumPy backend's regionprops validates them (only when
    # an object exists); keep that contract and message.
    if (
        prep.n > 0
        and pixels is not None
        and not (
            numpy.shape(pixels)[: labels.ndim] == labels.shape
            and numpy.ndim(pixels) in (labels.ndim, labels.ndim + 1)
        )
    ):
        raise ValueError(
            "Label and intensity image shapes must match, except for channel (last) axis."
        )
    (
        (min_y, min_x, max_y, max_x),
        count,
        center_y,
        center_x,
        filled,
        max_radius,
        mean_radius,
        median_radius,
    ) = _object_geometry(labels, prep.lut, prep.n, numpy_sum_chunk())
    # Same scalar expressions as skimage's regionprops (pixel area 1.0).
    area = count * 1.0
    area_bbox = ((max_y - min_y) * (max_x - min_x)) * 1.0
    equivalent_diameter = numpy.array([(2 * 2 * a / numpy.pi) ** (1 / 2) for a in area])

    raw, central, normalized, hu = spatial_moments_2d(labels, prep)
    area_convex = convex_area_2d(labels, prep)
    perimeter = perimeter_2d(labels, prep)
    crofton, euler = crofton_euler_2d(labels, prep)
    inertia = inertia_2d(central)
    axis_major, axis_minor, eccentricity, orientation = axes_eccentricity_orientation(
        inertia
    )

    formfactor = 4.0 * numpy.pi * area / perimeter**2
    denom = numpy.maximum(4.0 * numpy.pi * area, 1.0)
    compactness = perimeter**2 / denom
    solidity = area / area_convex

    results = {
        _ss.F_AREA: area,
        _ss.F_BBOX_AREA: area_bbox,
        _ss.F_CONVEX_AREA: area_convex,
        _ss.F_EQUIVALENT_DIAMETER: equivalent_diameter,
        _ss.F_PERIMETER: perimeter,
        _ss.F_MAJOR_AXIS_LENGTH: axis_major,
        _ss.F_MINOR_AXIS_LENGTH: axis_minor,
        _ss.F_ECCENTRICITY: eccentricity,
        _ss.F_ORIENTATION: orientation * (180 / numpy.pi),
        _ss.F_CENTER_X: center_x,
        _ss.F_CENTER_Y: center_y,
        _ss.F_MIN_X: min_x,
        _ss.F_MAX_X: max_x,
        _ss.F_MIN_Y: min_y,
        _ss.F_MAX_Y: max_y,
        _ss.F_FORM_FACTOR: formfactor,
        _ss.F_EXTENT: area / area_bbox,
        _ss.F_SOLIDITY: solidity,
        _ss.F_COMPACTNESS: compactness,
        _ss.F_EULER_NUMBER: euler,
        _ss.F_MAXIMUM_RADIUS: max_radius,
        _ss.F_MEAN_RADIUS: mean_radius,
        _ss.F_MEDIAN_RADIUS: median_radius,
    }
    if new_features:
        results |= {_ss.F_FILLED_AREA: filled * 1.0}

    if calculate_advanced:
        results |= moment_feature_dict(raw, central, normalized, hu, inertia)

    if new_features:
        results |= {_ss.F_PERIMETER_CROFTON: crofton}

    return results


def get_sizeshape(
    masks: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating] | None,
    calculate_advanced: bool = True,
    new_features: bool = True,
    spacing: tuple[float, ...] | None = None,
    *,
    max_workers: int | None = None,
) -> dict[str, NDArray[numpy.floating]] | list[dict[str, NDArray[numpy.floating]]]:
    """Size/shape backend for contiguous labels, with the NumPy public signature.

    2D and singleton-Z masks use the accelerated 2D measurements. Other inputs
    go unchanged to NumPy. Pixel shapes are validated even though no
    intensity-weighted measurements are requested. A batch (list/tuple or 4D
    array) returns one dict per image, measured on threads; ``max_workers`` caps
    the pool (default: the CPUs this process may use).
    """
    if is_batch(masks):
        return map_images(
            lambda item: _sizeshape_one(
                item[0], item[1], calculate_advanced, new_features, spacing
            ),
            per_image(masks, pixels),
            max_workers=max_workers,
            pixels=sum(numpy.size(m) for m in masks),
        )
    return _sizeshape_one(masks, pixels, calculate_advanced, new_features, spacing)


def _sizeshape_one(
    masks: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating] | None,
    calculate_advanced: bool,
    new_features: bool,
    spacing: tuple[float, ...] | None,
) -> dict[str, NDArray[numpy.floating]]:
    masks, pixels = _ss._single_slice_inputs(masks, pixels)
    if masks.ndim != 2:  # a single volume, so NumPy returns one dict
        return cast(
            dict[str, NDArray[numpy.floating]],
            _ss.get_sizeshape(masks, pixels, calculate_advanced, new_features, spacing),
        )
    return _sizeshape_2d(masks, pixels, calculate_advanced, new_features)
