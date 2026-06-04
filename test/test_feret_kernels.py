"""Unit tests for the numba feret boundary-ijv kernel.

The kernel must (1) emit boundary pixels in the SAME order as ``masks_to_ijv``
restricted to the boundary, and (2) yield bit-identical ``convex_hull_ijv`` and
``feret_diameter`` to the full-pixel path (interior pixels are never hull vertices).
"""

import centrosome.cpmorphology
import numpy as np
import pytest
from conftest import get_rng

from cp_measure._detect import HAS_NUMBA
from cp_measure.utils import masks_to_ijv

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


def _boundary_ref(masks):
    """Boundary pixels (8-connectivity) as masks_to_ijv would order them."""
    ijv = masks_to_ijv(masks)
    keep = []
    Y, X = masks.shape
    for i, j, v in ijv:
        is_b = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                y2, x2 = i + dy, j + dx
                if y2 < 0 or y2 >= Y or x2 < 0 or x2 >= X or masks[y2, x2] != v:
                    is_b = True
        keep.append(is_b)
    return ijv[np.array(keep, dtype=bool)] if len(keep) else ijv


def _rich_mask():
    """A 2D label image exercising squares, a disk, concave shapes, edges, contact."""
    m = np.zeros((80, 80), np.int32)
    m[2:20, 2:20] = 1  # square touching the top-left edge
    yy, xx = np.mgrid[0:80, 0:80]
    m[(yy - 40) ** 2 + (xx - 25) ** 2 <= 100] = 2  # disk
    m[10:40, 50:55] = 3  # thin bar
    m[50:70, 30:60] = 4  # rectangle
    m[55:65, 40:50] = 0  # punch a hole -> concave object 4
    m[70:75, 70:75] = 5  # small square touching bottom-right edge
    m[60, 70] = 6  # single-pixel object
    return m


@requires_numba
def test_boundary_ijv_matches_reference_order():
    from cp_measure.core.numba._feret import _boundary_ijv

    masks = _rich_mask()
    got, _ = _boundary_ijv(np.ascontiguousarray(masks), int(masks.max()))
    ref = _boundary_ref(masks)
    assert np.array_equal(got, ref)


@requires_numba
def test_boundary_ijv_yields_identical_hull_and_feret():
    from cp_measure.core.numba._feret import _boundary_ijv

    masks = _rich_mask()
    ijv_full = masks_to_ijv(masks)
    idx = np.unique(ijv_full[:, 2])
    idx = idx[idx > 0]
    ch_f, cc_f = centrosome.cpmorphology.convex_hull_ijv(ijv_full, idx)
    fmin_f, fmax_f = centrosome.cpmorphology.feret_diameter(ch_f, cc_f, idx)

    ijv_b, counts = _boundary_ijv(np.ascontiguousarray(masks), int(masks.max()))
    assert np.array_equal(np.nonzero(counts)[0], idx)  # counts recovers the labels
    ch_b, cc_b = centrosome.cpmorphology.convex_hull_ijv(ijv_b, idx)
    fmin_b, fmax_b = centrosome.cpmorphology.feret_diameter(ch_b, cc_b, idx)

    assert np.array_equal(ch_f, ch_b)
    assert np.array_equal(cc_f, cc_b)
    assert np.array_equal(fmin_f, fmin_b)
    assert np.array_equal(fmax_f, fmax_b)


@requires_numba
def test_boundary_ijv_random_masks():
    from cp_measure.core.numba._feret import _boundary_ijv

    rng = get_rng(7)
    for _ in range(5):
        # random blobby labels via nearest-centre assignment
        side, n = 48, 6
        yy, xx = np.mgrid[0:side, 0:side]
        centers = rng.integers(0, side, size=(n, 2))
        best = np.full((side, side), np.inf)
        masks = np.zeros((side, side), np.int32)
        for i, (cy, cx) in enumerate(centers, start=1):
            d = (yy - cy) ** 2 + (xx - cx) ** 2
            sel = d < best
            best[sel] = d[sel]
            masks[sel] = i
        got, _ = _boundary_ijv(np.ascontiguousarray(masks), int(masks.max()))
        assert np.array_equal(got, _boundary_ref(masks))


@requires_numba
def test_boundary_ijv_empty_mask():
    from cp_measure.core.numba._feret import _boundary_ijv

    masks = np.zeros((16, 16), np.int32)
    got, counts = _boundary_ijv(masks, 0)
    assert got.shape == (0, 3)
    assert counts.sum() == 0
