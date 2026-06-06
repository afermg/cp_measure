"""Golden tests: numba ``get_feret`` must match the numpy backend.

feret is 2D-only and ``pixels``-independent. The numba backend follows the
``to_bzyx`` batch convention (single image -> dict, list/4D -> list of dicts);
3D volumes yield ``{}`` like the baseline.
"""

import numpy as np
import pytest
from conftest import get_rng

from cp_measure._detect import HAS_NUMBA
from cp_measure.core.measureobjectsizeshape import get_feret as feret_numpy

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


def _rich_mask():
    m = np.zeros((80, 80), np.int32)
    m[2:20, 2:20] = 1
    yy, xx = np.mgrid[0:80, 0:80]
    m[(yy - 40) ** 2 + (xx - 25) ** 2 <= 100] = 2
    m[10:40, 50:55] = 3
    m[50:70, 30:60] = 4
    m[55:65, 40:50] = 0
    m[70:75, 70:75] = 5
    m[60, 70] = 6
    return m


def _assert_match(ref, got):
    assert set(got) == set(ref), set(got).symmetric_difference(ref)
    for key in ref:
        np.testing.assert_array_equal(got[key], ref[key], err_msg=f"feature {key!r}")


@requires_numba
def test_feret_matches_numpy_2d():
    from cp_measure.core.numba import get_feret as feret_nb

    masks = _rich_mask()
    pixels = get_rng().random(masks.shape)
    _assert_match(feret_numpy(masks, pixels), feret_nb(masks, pixels))


@requires_numba
def test_feret_pixels_none_matches_numpy():
    """feret is shape-only: the accelerator dispatch calls it with ``pixels=None`` (the
    baseline convention). The numba backend must accept None rather than handing it to
    ``to_bzyx`` (which rejects the 0-D ``numpy.asarray(None)``)."""
    from cp_measure.core.numba import get_feret as feret_nb

    masks = _rich_mask()
    _assert_match(feret_numpy(masks, None), feret_nb(masks, None))


@requires_numba
def test_feret_noncontiguous_labels_match_numpy():
    """Labels need not be 1..n contiguous; nonzero(counts) must recover the same
    ascending index set numpy gets from unique(ijv[:, 2])."""
    from cp_measure.core.numba import get_feret as feret_nb

    m = _rich_mask()
    # remap labels 1..6 -> {2, 5, 9, 11, 20, 30} (sparse, non-contiguous)
    remap = {1: 2, 2: 5, 3: 9, 4: 11, 5: 20, 6: 30}
    out = np.zeros_like(m)
    for src, dst in remap.items():
        out[m == src] = dst
    pixels = get_rng().random(out.shape)
    _assert_match(feret_numpy(out, pixels), feret_nb(out, pixels))


@requires_numba
def test_feret_bool_mask_matches_numpy():
    """The baseline accepts a single-object bool mask; the numba backend must too
    (numba cannot index an int array with a bool, so it casts internally)."""
    from cp_measure.core.numba import get_feret as feret_nb

    m = np.zeros((20, 20), dtype=bool)
    m[3:15, 4:12] = True
    pixels = get_rng().random(m.shape)
    _assert_match(feret_numpy(m, pixels), feret_nb(m, pixels))


@requires_numba
def test_feret_3d_returns_empty():
    from cp_measure.core.numba import get_feret as feret_nb

    masks = np.zeros((4, 16, 16), np.int32)
    masks[1:3, 4:10, 4:10] = 1
    pixels = get_rng().random(masks.shape)
    assert feret_nb(masks, pixels) == {}
    assert feret_numpy(masks, pixels) == {}


@requires_numba
def test_feret_empty_mask_matches_numpy_raises():
    """An all-background mask is unsupported by the baseline (convex_hull_ijv does
    np.max on empty input). The numba backend is a drop-in, so it raises the same
    ValueError rather than silently diverging."""
    from cp_measure.core.numba import get_feret as feret_nb

    masks = np.zeros((16, 16), np.int32)
    pixels = get_rng().random(masks.shape)
    with pytest.raises(ValueError):
        feret_numpy(masks, pixels)
    with pytest.raises(ValueError):
        feret_nb(masks, pixels)


@requires_numba
def test_feret_batch_matches_per_image_numpy():
    from cp_measure.core.numba import get_feret as feret_nb

    rng = get_rng(3)
    masks_list, pixels_list = [], []
    for _ in range(3):
        side, n = 48, 5
        yy, xx = np.mgrid[0:side, 0:side]
        centers = rng.integers(0, side, size=(n, 2))
        best = np.full((side, side), np.inf)
        m = np.zeros((side, side), np.int32)
        for i, (cy, cx) in enumerate(centers, start=1):
            d = (yy - cy) ** 2 + (xx - cx) ** 2
            sel = d < best
            best[sel] = d[sel]
            m[sel] = i
        masks_list.append(m)
        pixels_list.append(rng.random((side, side)))

    got = feret_nb(masks_list, pixels_list)
    assert isinstance(got, list) and len(got) == 3
    for m, p, g in zip(masks_list, pixels_list, got):
        _assert_match(feret_numpy(m, p), g)
