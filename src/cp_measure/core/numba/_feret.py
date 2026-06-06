"""Numba-backed MeasureObjectSizeShape Feret diameters (2D only).

Drop-in for :func:`cp_measure.core.measureobjectsizeshape.get_feret`. The numpy
baseline spends ~86% of its time in ``utils.masks_to_ijv`` — a per-label
``numpy.where`` scan that re-reads the whole image once per object — and feeds
*every* object pixel to ``centrosome.cpmorphology.convex_hull_ijv``.

Two bit-exact reductions collapse both costs into one numba pass:

1. **Replace the per-label scan.** A single row-major scatter into per-label
   offsets produces the SAME ``(i, j, label)`` rows in the SAME order as
   ``masks_to_ijv`` (label ascending, row-major within a label) — bit-identical.
2. **Feed the hull only boundary pixels.** The convex hull of an object equals
   the hull of its boundary: an interior pixel (all 8 neighbours share its label)
   can never be a hull vertex. Emitting only boundary pixels leaves
   ``convex_hull_ijv`` and ``feret_diameter`` bit-identical while shrinking the
   hull input ~17x (≈6% of pixels on typical masks). Boundary detection is a
   mechanical neighbour test, not numerically-sensitive geometry, so it stays on
   the reimplement side of the boundary rule.

``convex_hull_ijv`` / ``feret_diameter`` stay in centrosome (computational
geometry — imported). Serial kernel; no ``prange``/``nogil``. Batch via
``to_bzyx``; 3D volumes return ``{}`` like the baseline.
"""

import centrosome.cpmorphology
import numpy
from numba import njit
from numpy.typing import NDArray

from cp_measure.core.measureobjectsizeshape import (
    F_MAX_FERET_DIAMETER,
    F_MIN_FERET_DIAMETER,
)
from cp_measure.primitives.shapes import to_bzyx


@njit(cache=True)
def _boundary_ijv(masks, max_label):
    """Boundary-pixel ``(i, j, label)`` rows plus the per-label pixel ``counts``.

    A foreground pixel is on the boundary when any 8-neighbour (or the image edge)
    differs from its label. Rows are ``(n, 3)`` int64, label-ascending and
    row-major within a label — identical to ``masks_to_ijv`` restricted to
    boundary pixels. ``counts[label]`` (the second return) lets the caller recover
    the present labels without re-sorting. The expensive neighbour test runs once:
    pass 1 flags boundary pixels and counts per label, pass 2 scatters the flagged
    pixels into per-label offsets (mutated in place as the write cursor).
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
    return out, counts


def _feret_2d(masks_2d: NDArray[numpy.integer]) -> dict[str, NDArray[numpy.floating]]:
    masks_2d = numpy.ascontiguousarray(masks_2d)
    if (
        masks_2d.dtype == numpy.bool_
    ):  # baseline accepts a bool mask; numba can't index on it
        masks_2d = masks_2d.view(numpy.uint8)
    max_label = int(masks_2d.max()) if masks_2d.size else 0
    ijv, counts = _boundary_ijv(masks_2d, max_label)
    # counts[label] > 0 exactly for present labels; nonzero is ascending, no sort.
    indices = numpy.nonzero(counts)[0]
    chulls, chull_counts = centrosome.cpmorphology.convex_hull_ijv(ijv, indices)
    min_feret_diameter, max_feret_diameter = centrosome.cpmorphology.feret_diameter(
        chulls, chull_counts, indices
    )
    return {
        F_MIN_FERET_DIAMETER: min_feret_diameter,
        F_MAX_FERET_DIAMETER: max_feret_diameter,
    }


def get_feret(
    masks: NDArray[numpy.integer], pixels: NDArray[numpy.floating]
) -> dict[str, NDArray[numpy.floating]]:
    """Feret diameters (2D only). 3D volumes yield ``{}``, as in the baseline."""
    # Feret is shape-only and ignores ``pixels`` (the baseline accepts ``None``). Feed the
    # mask into the (masks, pixels) batch-normaliser's pixel slot so it has a real array to
    # shape-check against — ``to_bzyx`` rejects ``None`` (``numpy.asarray(None)`` is 0-D).
    masks_zyx, _pixels_zyx, unwrap = to_bzyx(masks, masks if pixels is None else pixels)
    results = [_feret_2d(m[0]) if m.shape[0] == 1 else {} for m in masks_zyx]
    return unwrap(results)
