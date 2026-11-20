"""Unit tests for the numba feret boundary-ijv kernel.

The kernel must (1) emit boundary pixels in the SAME order as ``masks_to_ijv``
restricted to the boundary, and (2) yield bit-identical ``convex_hull_ijv`` and
``feret_diameter`` to the full-pixel path (interior pixels are never hull vertices).
The Numba hull and diameters fed by it must equal centrosome's bit for bit.
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
    got = _boundary_ijv(np.ascontiguousarray(masks), int(masks.max()))
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

    ijv_b = _boundary_ijv(np.ascontiguousarray(masks), int(masks.max()))
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
        got = _boundary_ijv(np.ascontiguousarray(masks), int(masks.max()))
        assert np.array_equal(got, _boundary_ref(masks))


@requires_numba
def test_boundary_ijv_empty_mask():
    from cp_measure.core.numba._feret import _boundary_ijv

    masks = np.zeros((16, 16), np.int32)
    got = _boundary_ijv(masks, 0)
    assert got.shape == (0, 3)


def _centrosome_feret(masks):
    """Centrosome's hull and Feret diameters on every labelled pixel (NumPy path)."""
    idx = np.arange(1, int(masks.max()) + 1)
    chulls, counts = centrosome.cpmorphology.convex_hull_ijv(masks_to_ijv(masks), idx)
    return centrosome.cpmorphology.feret_diameter(chulls, counts, idx)


def _numba_feret(masks):
    from cp_measure.core.numba._feret import _boundary_ijv, _feret_ijv

    masks = np.ascontiguousarray(masks)
    return _feret_ijv(_boundary_ijv(masks, int(masks.max())), int(masks.max()))


def _stamp(*objects):
    """A 12x12 mask with one label per ``(rows, cols)`` slice pair (0 = absent)."""
    m = np.zeros((12, 12), np.int32)
    for label, region in enumerate(objects, start=1):
        if region is not None:
            m[region] = label
    return m


_DIAG = (np.arange(2, 7), np.arange(3, 8))
_SHAPES = {
    "rich": _rich_mask(),
    "single_pixel": _stamp((5, 5)),
    "two_pixels": _stamp((1, slice(1, 3)), (slice(4, 6), 1), ([8, 9], [8, 9])),
    "collinear": _stamp(
        (1, slice(0, 9)), (slice(3, 11), 2), _DIAG, (_DIAG[0], _DIAG[1][::-1] + 2)
    ),
    # Centrosome's antipodal walk skips a pair on squares; the port must too.
    "squares": _stamp(
        (slice(0, 2), slice(0, 2)),
        (slice(3, 6), slice(0, 3)),
        (slice(7, 11), slice(4, 12)),
    ),
    "absent_labels": _stamp(None, (slice(2, 5), slice(2, 9)), None, (8, 8), None),
}


@requires_numba
@pytest.mark.parametrize("name", list(_SHAPES))
def test_feret_matches_centrosome(name):
    """Diameters are bit-identical to centrosome; absent labels give 0."""
    masks = _SHAPES[name]
    for got, ref in zip(_numba_feret(masks), _centrosome_feret(masks)):
        np.testing.assert_array_equal(got, ref)


@requires_numba
def test_feret_matches_centrosome_random_labels():
    """Scattered (disconnected) and blobby random labels, many small hulls."""
    rng = get_rng(11)
    for trial in range(300):
        side = int(rng.integers(3, 32))
        n = int(rng.integers(1, 8))
        masks = rng.integers(0, n + 1, size=(side, side)).astype(np.int32)
        if trial % 2:  # blobs: each pixel takes its nearest centre's label
            yy, xx = np.mgrid[0:side, 0:side]
            c = rng.integers(0, side, size=(n, 2))
            d = (yy[..., None] - c[:, 0]) ** 2 + (xx[..., None] - c[:, 1]) ** 2
            masks = (d.argmin(-1) + 1).astype(np.int32)
        masks[rng.random((side, side)) < rng.random()] = 0
        if masks.max() == 0:
            continue
        for got, ref in zip(_numba_feret(masks), _centrosome_feret(masks)):
            np.testing.assert_array_equal(got, ref)
