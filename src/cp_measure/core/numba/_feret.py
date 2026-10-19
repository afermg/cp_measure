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
order within each label. Centrosome still computes the hull and diameters; only
the input preparation changes. Larger solid objects usually benefit most because
a smaller fraction of their pixels lie on the boundary.

The kernel is serial. A batch is a list/tuple of images or a 4D ``(B, Z, Y, X)``
array. Like raw NumPy Feret, 3D volumes return ``{}``.
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
    # Numba rejects non-native-endian arrays. Convert here so raw and bulk callers
    # accept the same byte orders as NumPy, and give the kernel contiguous storage.
    masks_2d = numpy.ascontiguousarray(masks_2d, dtype=masks_2d.dtype.newbyteorder("="))
    max_label = int(masks_2d.max()) if masks_2d.size else 0
    ijv = _boundary_ijv(masks_2d, max_label)
    # The featurizer and bulk wrapper relabel objects to 1..N before this call.
    # Raw callers must supply contiguous labels too.
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
    """Measure a 2D mask; return {} for a 3D volume, matching raw NumPy Feret.

    A single-slice (1, Y, X) volume is still 3D. Keep this compatibility behavior
    here; the featurizer already excludes Feret when measuring volumes.
    """
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
