"""Numba-backed MeasureObjectSizeShape Feret diameters (2D only).

Feret diameters are the minimum and maximum distances between parallel tangent
lines wrapped around an object, over every orientation. We find the object's
convex hull — like a rubber band stretched around its outside — then read the
smallest and largest gap between opposite sides of that hull.

Drop-in for :func:`cp_measure.core.measureobjectsizeshape.get_feret`. On current
main the numpy baseline is dominated by ``centrosome.cpmorphology.convex_hull_ijv``
(~75% of ``get_feret`` at 1080²/144 obj), which is fed *every* object pixel;
``utils.masks_to_ijv`` is already a fast vectorised ``nonzero`` + stable
``argsort``. So the win here is shrinking the hull input, in one numba pass:

1. **Reconstruct the ijv order.** A single row-major scatter into per-label
   offsets produces the SAME ``(i, j, label)`` rows in the SAME order as
   ``masks_to_ijv`` (label ascending, row-major within a label). This matches the
   numpy path bit-for-bit — it is not itself a speedup, it just emits the boundary
   pixels in the order ``convex_hull_ijv`` expects.
2. **Feed the hull only boundary pixels.** The convex hull of an object equals
   the hull of its boundary: an interior pixel (all 8 neighbours share its label)
   can never be a hull vertex. Emitting only boundary pixels leaves
   ``convex_hull_ijv`` and ``feret_diameter`` bit-identical while shrinking the
   hull input ~17x (≈6% of pixels on typical masks) — this is the actual speedup.
   Boundary detection is a mechanical neighbour test, not numerically-sensitive
   geometry, so it stays on the reimplement side of the boundary rule.

``convex_hull_ijv`` / ``feret_diameter`` stay in centrosome (computational
geometry — imported). Serial kernel; no ``prange``/``nogil``. A batch is a
list/tuple of images or a 4D ``(B, Z, Y, X)`` array; 3D volumes return ``{}``
like the baseline.
"""

import centrosome.cpmorphology
import numpy
from numba import njit
from numpy.typing import NDArray

from cp_measure.core.measureobjectsizeshape import (
    F_MAX_FERET_DIAMETER,
    F_MIN_FERET_DIAMETER,
)
from cp_measure.primitives.shapes import _stack


@njit(cache=True)
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


def _feret_2d(masks_2d: NDArray[numpy.integer]) -> dict[str, NDArray[numpy.floating]]:
    masks_2d = numpy.asarray(masks_2d)
    if (
        masks_2d.dtype == numpy.bool_
    ):  # baseline accepts a bool mask; numba can't index on it
        masks_2d = masks_2d.view(numpy.uint8)
    # numba reads the raw buffer, so force native byte order: a non-native dtype
    # (e.g. ">i4") is valid in numpy but numba would misread it. This also makes
    # the array contiguous, as the kernel requires.
    masks_2d = numpy.ascontiguousarray(masks_2d, dtype=masks_2d.dtype.newbyteorder("="))
    max_label = int(masks_2d.max()) if masks_2d.size else 0
    ijv = _boundary_ijv(masks_2d, max_label)
    # Labels are contiguous 1..N by contract (sanitized at the entry points), so the
    # hull indices are exactly the baseline's ``arange`` — bit-for-bit, and the
    # backend doesn't quietly diverge on gapped labels it was never meant to accept.
    indices = numpy.arange(1, max_label + 1)
    chulls, chull_counts = centrosome.cpmorphology.convex_hull_ijv(ijv, indices)
    min_feret_diameter, max_feret_diameter = centrosome.cpmorphology.feret_diameter(
        chulls, chull_counts, indices
    )
    return {
        F_MIN_FERET_DIAMETER: min_feret_diameter,
        F_MAX_FERET_DIAMETER: max_feret_diameter,
    }


def _feret_image(
    masks: NDArray[numpy.integer],
) -> dict[str, NDArray[numpy.floating]]:
    """One image: 2D -> Feret; any 3D volume (incl. single-slice ``(1, Y, X)``) ->
    ``{}``, keyed on ``ndim`` exactly as the baseline, which never normalises a 2D
    image to a volume."""
    return {} if numpy.ndim(masks) == 3 else _feret_2d(masks)


def get_feret(
    masks: NDArray[numpy.integer], pixels: NDArray[numpy.floating] | None = None
) -> dict[str, NDArray[numpy.floating]] | list[dict[str, NDArray[numpy.floating]]]:
    """Feret diameters (2D only); a single image, or a batch -> list of dicts.

    A batch is a list/tuple of images or a 4D ``(B, Z, Y, X)`` array; 3D volumes
    yield ``{}``, as in the baseline. ``pixels`` is ignored — Feret is shape-only
    — and kept only to match the ``(masks, pixels)`` measurement signature. Images
    are dispatched by ``ndim`` rather than through ``to_bzyx``: normalising would
    make a 2D image indistinguishable from a single-slice volume, which the
    baseline measures differently (Feret vs ``{}``).

    So a batch of 2D images must be a *list* of ``(Y, X)`` arrays: a ``(B, 1, Y, X)``
    array is a batch of single-slice *volumes* and yields ``[{}, ...]``.
    """
    masks_batched = isinstance(masks, (list, tuple)) or numpy.ndim(masks) == 4
    if not masks_batched:
        return _feret_image(masks)
    if isinstance(masks, (list, tuple)):
        masks = _stack(masks, "masks")
    return [_feret_image(m) for m in masks]
