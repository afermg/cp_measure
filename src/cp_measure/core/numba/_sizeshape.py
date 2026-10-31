"""Numba 2D size/shape geometry, isolated from PR #78's integration-stack base.

Accelerates convex hull construction and label-aware perimeter/topology reductions.
Moments and their derived features retain the NumPy backend's batched products:
changing their reduction order can flip orientation at the +/-90 degree boundary.
The skimage hull rasterizer and scipy Euclidean distance transform remain on host.
"""

from typing import NamedTuple

import numba
import numpy
import scipy.ndimage
from numpy.typing import NDArray
from skimage.measure import grid_points_in_poly, regionprops_table

from cp_measure.core import measureobjectsizeshape as _ss
from cp_measure.primitives._moments import (
    axes_eccentricity_orientation,
    inertia_2d,
    moment_feature_dict,
    spatial_moments_2d,
)


class _Prep(NamedTuple):
    """Label lookup and foreground counts shared by the geometry kernels."""

    lut: NDArray[numpy.int64]
    n: int
    offsets: NDArray[numpy.int64]


def _foreground_prep(labels: NDArray[numpy.integer]) -> _Prep:
    """One bincount pass; geometry primitives also accept sparse positive labels."""
    counts = numpy.bincount(labels.ravel().astype(numpy.intp, copy=False))
    present = numpy.nonzero(counts[1:])[0] + 1
    n = int(present.size)
    lut = numpy.full(len(counts), -1, dtype=numpy.int64)
    lut[present] = numpy.arange(n, dtype=numpy.int64)
    offsets = numpy.zeros(n + 1, dtype=numpy.int64)
    offsets[1:] = numpy.cumsum(counts[present])
    return _Prep(lut, n, offsets)


# --- Convex hull (area_convex / solidity) --------------------------------------------------
# skimage's `area_convex` is the pixel count inside the convex hull of each object's pixels,
# where each pixel contributes its 4 edge-midpoints (±0.5 "diamond" offsets) and the count comes
# from `grid_points_in_poly`. We replace only the slow per-region hull *construction* (scipy
# QHull) with a fused numba monotone-chain over each object's BOUNDARY pixels (hull(boundary) ==
# hull(object); ~6-17x fewer points), and keep skimage's exact `grid_points_in_poly` raster — so
# the area retains skimage's pixel-inclusion convention. Coordinates are scaled x2 so offsets are
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
    order = numpy.argsort(obj4, kind="stable")
    px, py, obj4 = px[order], py[order], obj4[order]
    grp = numpy.zeros(n + 1, numpy.int64)
    numpy.add.at(grp, obj4 + 1, 1)
    numpy.cumsum(grp, out=grp)
    stride = 4 * max(labels.shape) + 10
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


@numba.njit(cache=True)
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


@numba.njit(cache=True)
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
    return _perimeter_kernel(
        numpy.ascontiguousarray(labels, dtype=numpy.int64), lut, n, _PERIMETER_WEIGHTS
    )


def crofton_euler_2d(
    labels: NDArray[numpy.integer], prep: _Prep | None = None
) -> tuple[NDArray[numpy.floating], NDArray[numpy.int64]]:
    """Per-object ``(perimeter_crofton, euler_number)`` from the shared 2x2-config histogram."""
    if prep is None:
        prep = _foreground_prep(labels)
    lut, n = prep.lut, prep.n
    if n == 0:
        return numpy.zeros(0), numpy.zeros(0, dtype=numpy.int64)
    padded = numpy.pad(numpy.ascontiguousarray(labels, dtype=numpy.int64), 1)
    hist = _xf_hist_kernel(padded, lut, n)
    return hist @ _CROFTON_COEFS_4, hist @ _EULER_COEFS_8


# --- Full get_sizeshape wrapper -------------------------------------------------------------
# Assembles the complete sizeshape feature dict, sourcing QHull / per-region geometry
# pieces from the numba kernels above and keeping only cheap, moment-free regionprops props
# (area / bbox / centroid / extent / area_filled / image) plus the scipy Euclidean EDT radius
# loop. With option B (axes/eccentricity/orientation derived from the central moments),
# regionprops computes no moments at all.

# moment-free regionprops props (verified 0 ms einsum); `image` feeds the EDT radius loop.
_CHEAP_PROPS = [
    "image",
    "area",
    "area_bbox",
    "equivalent_diameter_area",
    "bbox",
    "centroid",
    "extent",
]


def _sizeshape_2d(
    labels: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating] | None,
    calculate_advanced: bool,
    new_features: bool,
) -> dict[str, NDArray]:
    props_list = _CHEAP_PROPS + (["area_filled"] if new_features else [])
    props = regionprops_table(labels, pixels, properties=props_list)
    area = props["area"]

    prep = _foreground_prep(labels)
    raw, central, normalized, hu = spatial_moments_2d(labels)
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

    nobjects = len(props["image"])
    max_radius = numpy.zeros(nobjects)
    mean_radius = numpy.zeros(nobjects)
    median_radius = numpy.zeros(nobjects)
    for index, mini_image in enumerate(props["image"]):
        mini_image = numpy.pad(mini_image, 1)
        distances = scipy.ndimage.distance_transform_edt(mini_image)
        inside = distances[mini_image]
        max_radius[index] = inside.max()
        mean_radius[index] = inside.mean()
        median_radius[index] = numpy.median(inside)

    results = {
        _ss.F_AREA: area,
        _ss.F_BBOX_AREA: props["area_bbox"],
        _ss.F_CONVEX_AREA: area_convex,
        _ss.F_EQUIVALENT_DIAMETER: props["equivalent_diameter_area"],
        _ss.F_PERIMETER: perimeter,
        _ss.F_MAJOR_AXIS_LENGTH: axis_major,
        _ss.F_MINOR_AXIS_LENGTH: axis_minor,
        _ss.F_ECCENTRICITY: eccentricity,
        _ss.F_ORIENTATION: orientation * (180 / numpy.pi),
        _ss.F_CENTER_X: props["centroid-1"],
        _ss.F_CENTER_Y: props["centroid-0"],
        _ss.F_MIN_X: props["bbox-1"],
        _ss.F_MAX_X: props["bbox-3"],
        _ss.F_MIN_Y: props["bbox-0"],
        _ss.F_MAX_Y: props["bbox-2"],
        _ss.F_FORM_FACTOR: formfactor,
        _ss.F_EXTENT: props["extent"],
        _ss.F_SOLIDITY: solidity,
        _ss.F_COMPACTNESS: compactness,
        _ss.F_EULER_NUMBER: euler,
        _ss.F_MAXIMUM_RADIUS: max_radius,
        _ss.F_MEAN_RADIUS: mean_radius,
        _ss.F_MEDIAN_RADIUS: median_radius,
    }
    if new_features:
        results |= {_ss.F_FILLED_AREA: props["area_filled"]}

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
) -> dict[str, NDArray[numpy.floating]]:
    """Size/shape backend for contiguous labels, with the NumPy public signature.

    Only 2D masks are accelerated. All other ranks (including singleton-Z) go
    unchanged to NumPy, preserving its dimensional support and error behavior.
    ``pixels`` is forwarded for regionprops' shape validation even though no
    intensity-weighted measurements are requested.
    """
    if masks.ndim != 2:
        return _ss.get_sizeshape(
            masks, pixels, calculate_advanced, new_features, spacing
        )
    return _sizeshape_2d(masks, pixels, calculate_advanced, new_features)
