"""Numba-backed MeasureObjectSizeShape Feret diameters (2D only).

Feret diameters are the minimum and maximum distances between parallel tangent
lines wrapped around an object, over every orientation. We find the object's
convex hull — like a rubber band stretched around its outside — then read the
smallest and largest gap between opposite sides of that hull.

Replaces :func:`cp_measure.core.measureobjectsizeshape.get_feret` when the Numba
accelerator is selected. The NumPy implementation sends every object pixel to
centrosome's convex-hull routine. This implementation sends only boundary pixels:
removing interior pixels cannot change the convex hull.

The Numba kernel marks pixels touching a different label or the image edge,
then emits their ``(row, column, label)`` coordinates in label order and row-major
order within each label. Numba ports of centrosome's ``convex_hull_ijv`` and
``feret_diameter`` then compute the hull and diameters with the same arithmetic,
vertex order and tie-breaking, so results are bit-identical to NumPy.

The kernels are serial and release the GIL; a batch (a list/tuple of images or a
4D ``(B, Z, Y, X)`` array) measures its images on threads. Like raw NumPy Feret,
3D volumes return ``{}``.
"""

import numpy
from numba import njit
from numpy.typing import NDArray

from cp_measure.core.measureobjectsizeshape import (
    F_MAX_FERET_DIAMETER,
    F_MIN_FERET_DIAMETER,
)
from cp_measure.primitives.parallel import map_images
from cp_measure.primitives.shapes import as_2d, is_batch, per_image


@njit(cache=True, nogil=True)
def _boundary_ijv(masks, max_label):
    """Boundary-pixel ``(i, j, label)`` rows, ``(n, 3)`` int64.

    A foreground pixel is on the boundary when any 8-neighbour (or the image edge)
    differs from its label. Rows are label-ascending and row-major within a label
    — identical to ``masks_to_ijv`` restricted to boundary pixels. The expensive
    neighbour test runs once: pass 1 flags boundary pixels and counts per label,
    pass 2 scatters the flagged pixels into per-label offsets (mutated in place as
    the write cursor).
    """
    Y, X = masks.shape
    is_b = numpy.zeros((Y, X), numpy.bool_)
    counts = numpy.zeros(max_label + 1, numpy.int64)
    for y in range(Y):
        for x in range(X):
            v = masks[y, x]
            if v <= 0:
                continue
            boundary = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    y2 = y + dy
                    x2 = x + dx
                    if y2 < 0 or y2 >= Y or x2 < 0 or x2 >= X or masks[y2, x2] != v:
                        boundary = True
                        break
                if boundary:
                    break
            if boundary:
                is_b[y, x] = True
                counts[v] += 1

    # offs[lbl] = start row for label lbl; offs[max_label+1] = total boundary pixels.
    offs = numpy.zeros(max_label + 2, numpy.int64)
    for lbl in range(1, max_label + 1):
        offs[lbl + 1] = offs[lbl] + counts[lbl]
    out = numpy.empty((offs[max_label + 1], 3), numpy.int64)
    for y in range(Y):
        for x in range(X):
            if is_b[y, x]:
                v = masks[y, x]
                p = offs[v]  # offs[1..max_label] doubles as the write cursor here
                out[p, 0] = y
                out[p, 1] = x
                out[p, 2] = v
                offs[v] = p + 1
    return out


def _boundary(masks: NDArray[numpy.integer]):
    """``(ijv, max_label)`` of 2D or singleton-Z masks; ``None`` for other volumes,
    which NumPy skips."""
    masks_2d = as_2d(masks)
    if masks_2d is None:
        return None
    masks_2d = numpy.asarray(masks_2d)
    if (
        masks_2d.dtype == numpy.bool_
    ):  # baseline accepts a bool mask; numba can't index on it
        masks_2d = masks_2d.view(numpy.uint8)
    # Numba rejects non-native-endian arrays. Convert here so raw and bulk callers
    # accept the same byte orders as NumPy, and give the kernel contiguous storage.
    masks_2d = numpy.ascontiguousarray(masks_2d, dtype=masks_2d.dtype.newbyteorder("="))
    max_label = int(masks_2d.max()) if masks_2d.size else 0
    return _boundary_ijv(masks_2d, max_label), max_label


@njit(cache=True, nogil=True)
def _convex(a_i, a_j, b_i, b_j, c_i, c_j):
    """Centrosome's ``CONVEX``: does the path a -> b -> c turn left?"""
    cross = (b_j - a_j) * (c_i - b_i) - (c_j - b_j) * (b_i - a_i)
    if cross > 0:
        return True
    if cross < 0:
        return False
    # Collinear: only a U-turn at the rightmost column counts as convex.
    return b_j > a_j and b_j > c_j


@njit(cache=True, nogil=True)
def _convex_hulls(ijv, max_label):
    """Centrosome's ``_convex_hull.convex_hull_ijv`` for labels ``1..max_label``.

    Returns ``(hull, offsets)``: label ``v``'s vertices are
    ``hull[offsets[v - 1]:offsets[v]]`` as ``(i, j)`` rows, in centrosome's order.
    Centrosome writes each hull over its own sorted input, so a vertex is dropped
    when the write would pass the end of that label's pixels. Output positions
    here follow the same running offsets so that cap applies identically. The
    row order within a label does not matter: only per-column extremes are used.
    """
    n = ijv.shape[0]
    counts = numpy.zeros(max_label + 1, numpy.int64)
    first_j = numpy.full(max_label + 1, numpy.iinfo(numpy.int64).max)
    last_j = numpy.full(max_label + 1, -1)
    max_i = 0
    max_j = 0
    for p in range(n):
        i, j, v = ijv[p, 0], ijv[p, 1], ijv[p, 2]
        counts[v] += 1
        first_j[v] = min(first_j[v], j)
        last_j[v] = max(last_j[v], j)
        max_i = max(max_i, i)
        max_j = max(max_j, j)

    no_lower = max_i + 1
    upper = numpy.empty(max_j + 1, numpy.int64)
    lower = numpy.empty(max_j + 1, numpy.int64)
    # One spare row: centrosome's last write per label is not capped.
    hull = numpy.empty((n + 1, 2), numpy.int64)
    offsets = numpy.zeros(max_label + 1, numpy.int64)
    label_start = 0
    out = 0
    for v in range(1, max_label + 1):
        label_end = label_start + counts[v]
        if counts[v] > 0:
            start_j = first_j[v]
            end_j = last_j[v]
            upper[start_j : end_j + 1] = -1
            lower[start_j : end_j + 1] = no_lower
            for p in range(label_start, label_end):
                i, j = ijv[p, 0], ijv[p, 1]
                upper[j] = max(upper[j], i)
                lower[j] = min(lower[j], i)

            need_last_upper = lower[start_j] != upper[start_j]
            k = 0
            # Lower envelope left to right, then upper envelope right to left,
            # popping vertices that are not convex.
            for j in range(start_j, end_j + 1):
                if lower[j] < no_lower:
                    while k >= 2 and not _convex(
                        hull[out + k - 2, 0],
                        hull[out + k - 2, 1],
                        hull[out + k - 1, 0],
                        hull[out + k - 1, 1],
                        lower[j],
                        j,
                    ):
                        k -= 1
                    hull[out + k, 0] = lower[j]
                    hull[out + k, 1] = j
                    k += 1
            for j in range(end_j, start_j, -1):
                if upper[j] > -1:
                    while k >= 2 and not _convex(
                        hull[out + k - 2, 0],
                        hull[out + k - 2, 1],
                        hull[out + k - 1, 0],
                        hull[out + k - 1, 1],
                        upper[j],
                        j,
                    ):
                        k -= 1
                    if label_end > out + k:
                        hull[out + k, 0] = upper[j]
                        hull[out + k, 1] = j
                        k += 1
            while k >= 2 and not _convex(
                hull[out + k - 2, 0],
                hull[out + k - 2, 1],
                hull[out + k - 1, 0],
                hull[out + k - 1, 1],
                upper[start_j],
                start_j,
            ):
                k -= 1
            if need_last_upper:
                hull[out + k, 0] = upper[start_j]
                hull[out + k, 1] = start_j
                k += 1
            out += k
        offsets[v] = out
        label_start = label_end
    return hull, offsets


@njit(cache=True, nogil=True, error_model="numpy")
def _distance2_to_line(p_i, p_j, a_i, a_j, b_i, b_j):
    """Centrosome's ``distance2_to_line``: squared distance from p to line a-b.

    Same float64 operations, so the result is bit-identical. A zero-length line
    gives NaN, as in NumPy.
    """
    cross = (a_i - b_i) * (a_j - p_j) - (a_i - p_i) * (a_j - b_j)
    d_i = b_i - a_i
    d_j = b_j - a_j
    return cross * cross / (d_i * d_i + d_j * d_j)


@njit(cache=True, nogil=True, error_model="numpy")
def _hull_feret(pts):
    """Squared (min, max) Feret diameter of one hull, as centrosome's
    ``feret_diameter``.

    Centrosome's antipodal-pair walk (Alsuwaiyel) starts at vertex 0 and the
    vertex farthest from the closing edge (first one on ties), then advances
    whichever of vertex or antipode keeps the antipode at least as far from the
    current edge. The walk can miss pairs (for a square it skips 1-2); this is
    kept. Max is over the walk's pairs. Min is the distance from vertex ``f`` to
    the edge between two of its consecutive antipodes. NaN distances are skipped
    and a hull with no candidate gets 0, as ``scipy.ndimage.minimum`` does.
    """
    m = pts.shape[0]
    if m < 2:
        return 0.0, 0.0
    if m == 2:
        d_i = pts[1, 0] - pts[0, 0]
        d_j = pts[1, 1] - pts[0, 1]
        return 0.0, d_i * d_i + d_j * d_j

    last = m - 1
    antipode = 1
    best = _distance2_to_line(
        pts[1, 0], pts[1, 1], pts[last, 0], pts[last, 1], pts[0, 0], pts[0, 1]
    )
    for k in range(2, last):
        d = _distance2_to_line(
            pts[k, 0], pts[k, 1], pts[last, 0], pts[last, 1], pts[0, 0], pts[0, 1]
        )
        if d > best or (numpy.isnan(best) and not numpy.isnan(d)):
            best = d
            antipode = k

    # Each pair is stored as first * (m + 1) + second, both ways round. A pair
    # whose second vertex is 0 is also stored with second == m, so that the
    # antipodes m - 1 and 0 count as consecutive.
    keys = numpy.empty(6 * m, numpy.int64)
    n_keys = 0
    max_d2 = 0.0
    vertex = 0
    while True:
        d_i = pts[antipode, 0] - pts[vertex, 0]
        d_j = pts[antipode, 1] - pts[vertex, 1]
        max_d2 = max(max_d2, d_i * d_i + d_j * d_j)
        keys[n_keys] = vertex * (m + 1) + antipode
        keys[n_keys + 1] = antipode * (m + 1) + vertex
        n_keys += 2
        if vertex == 0:
            keys[n_keys] = antipode * (m + 1) + m
            n_keys += 1

        nxt = antipode + 1 if antipode + 1 < m else 0
        a_i, a_j = pts[vertex, 0], pts[vertex, 1]
        b_i, b_j = pts[vertex + 1, 0], pts[vertex + 1, 1]
        d_cur = _distance2_to_line(
            pts[antipode, 0], pts[antipode, 1], a_i, a_j, b_i, b_j
        )
        d_next = _distance2_to_line(pts[nxt, 0], pts[nxt, 1], a_i, a_j, b_i, b_j)
        if d_cur <= d_next:
            antipode += 1
        else:
            vertex += 1
        if antipode >= m or vertex == antipode:
            break

    keys = numpy.sort(keys[:n_keys])
    min_d2 = numpy.nan
    for k in range(n_keys - 1):
        second = keys[k] % (m + 1)
        if second == m or keys[k + 1] != keys[k] + 1:
            continue
        first = keys[k] // (m + 1)
        after = second + 1 if second + 1 < m else 0
        d = _distance2_to_line(
            pts[first, 0],
            pts[first, 1],
            pts[second, 0],
            pts[second, 1],
            pts[after, 0],
            pts[after, 1],
        )
        if d < min_d2 or (numpy.isnan(min_d2) and not numpy.isnan(d)):
            min_d2 = d
    if numpy.isnan(min_d2):
        min_d2 = 0.0
    return min_d2, max_d2


@njit(cache=True, nogil=True)
def _feret_ijv(ijv, max_label):
    """(min, max) Feret diameters of labels ``1..max_label`` from boundary rows."""
    hull, offsets = _convex_hulls(ijv, max_label)
    pts = hull.astype(numpy.float64)
    min_feret = numpy.zeros(max_label)
    max_feret = numpy.zeros(max_label)
    for v in range(1, max_label + 1):
        min_d2, max_d2 = _hull_feret(pts[offsets[v - 1] : offsets[v]])
        min_feret[v - 1] = numpy.sqrt(min_d2)
        max_feret[v - 1] = numpy.sqrt(max_d2)
    return min_feret, max_feret


def _diameters(boundary) -> dict[str, NDArray[numpy.floating]]:
    """Feret diameters from :func:`_boundary`'s output (``{}`` for ``None``)."""
    if boundary is None:
        return {}
    ijv, max_label = boundary
    if max_label == 0:
        # Centrosome's hull cannot reduce an empty pixel list; keep its error.
        raise ValueError("Feret diameters need at least one labelled pixel")
    # The featurizer and bulk wrapper relabel objects to 1..N before this call.
    # Raw callers must supply contiguous labels too; absent labels measure 0.
    min_feret, max_feret = _feret_ijv(ijv, max_label)
    return {
        F_MIN_FERET_DIAMETER: min_feret,
        F_MAX_FERET_DIAMETER: max_feret,
    }


def get_feret(
    masks: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating] | None = None,
    *,
    max_workers: int | None = None,
) -> dict[str, NDArray[numpy.floating]] | list[dict[str, NDArray[numpy.floating]]]:
    """Feret diameters (2D only); a single image, or a batch -> list of dicts.

    A batch is a list/tuple of images or a 4D ``(B, Z, Y, X)`` array. Singleton-Z
    volumes use the 2D measurements; other volumes yield ``{}``, as in NumPy.
    ``pixels`` is ignored — Feret is shape-only — and kept only to match the
    ``(masks, pixels)`` measurement signature. The same Z-depth rule applies to
    individual inputs and every batch entry. ``max_workers`` caps the batch thread
    pool (default: the CPUs this process may use).
    """
    if not is_batch(masks):
        return _diameters(_boundary(masks))
    if len(masks) == 0:  # an empty batch measures nothing (each_volume rejects it)
        return []
    return map_images(
        lambda item: _diameters(_boundary(item[0])),
        per_image(masks, None),
        max_workers=max_workers,
        pixels=sum(numpy.size(m) for m in masks),
    )
