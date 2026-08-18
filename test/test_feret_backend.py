"""Golden tests: numba ``get_feret`` must match the numpy backend.

feret is 2D-only and ``pixels``-independent. The numba backend dispatches by
``ndim`` (single image -> dict, list/4D batch -> list of dicts); a 3D volume,
including a single-slice ``(1, Y, X)``, yields ``{}`` like the baseline.
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
    from cp_measure.core.numba._feret import get_feret as feret_nb

    masks = _rich_mask()
    pixels = get_rng().random(masks.shape)
    _assert_match(feret_numpy(masks, pixels), feret_nb(masks, pixels))


@requires_numba
def test_feret_pixels_none_matches_numpy():
    """feret is shape-only: the accelerator dispatch calls it with ``pixels=None`` (the
    baseline convention). The numba backend must accept None — it ignores ``pixels``
    entirely rather than feeding it to a shape-checker."""
    from cp_measure.core.numba._feret import get_feret as feret_nb

    masks = _rich_mask()
    _assert_match(feret_numpy(masks, None), feret_nb(masks, None))


@requires_numba
def test_feret_bool_mask_matches_numpy():
    """The baseline accepts a single-object bool mask; the numba backend must too
    (numba cannot index an int array with a bool, so it casts internally)."""
    from cp_measure.core.numba._feret import get_feret as feret_nb

    m = np.zeros((20, 20), dtype=bool)
    m[3:15, 4:12] = True
    pixels = get_rng().random(m.shape)
    _assert_match(feret_numpy(m, pixels), feret_nb(m, pixels))


@requires_numba
def test_feret_3d_returns_empty():
    from cp_measure.core.numba._feret import get_feret as feret_nb

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
    from cp_measure.core.numba._feret import get_feret as feret_nb

    masks = np.zeros((16, 16), np.int32)
    pixels = get_rng().random(masks.shape)
    with pytest.raises(ValueError):
        feret_numpy(masks, pixels)
    with pytest.raises(ValueError):
        feret_nb(masks, pixels)


@requires_numba
def test_feret_batch_matches_per_image_numpy():
    from cp_measure.core.numba._feret import get_feret as feret_nb

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


@requires_numba
def test_feret_single_slice_volume_returns_empty():
    """A ``(1, Y, X)`` array is a single-slice volume (ndim 3), not a 2D image, so
    both backends return ``{}``. Dispatching by ``ndim`` (not by normalising to
    ``(Z, Y, X)`` first) is what keeps numba from measuring it as 2D."""
    from cp_measure.core.numba._feret import get_feret as feret_nb

    masks = _rich_mask()[np.newaxis]  # (1, 80, 80)
    pixels = get_rng().random(masks.shape)
    assert feret_nb(masks, pixels) == {}
    assert feret_numpy(masks, pixels) == {}


@requires_numba
def test_feret_non_native_byteorder_matches_numpy():
    """A non-native-endian integer mask (e.g. ``>i4``) is valid in numpy; numba
    reads the raw buffer, so the backend normalises byte order to stay a drop-in."""
    from cp_measure.core.numba._feret import get_feret as feret_nb

    masks = _rich_mask().astype(">i4")
    pixels = get_rng().random(masks.shape)
    _assert_match(feret_numpy(masks, pixels), feret_nb(masks, pixels))


@requires_numba
def test_feret_through_accelerator_sanitizes_batch():
    """The real seam: ``set_accelerator("numba")`` wraps feret in ``sanitize``, which
    relabels each image of a list batch (PR #99) before the backend sees it. Exercises
    the full path a user hits, not the bare backend call."""
    import cp_measure.bulk  # binds cp_measure.bulk; not implied by `import cp_measure`

    masks_list = [_rich_mask(), _rich_mask()]
    pixels_list = [get_rng().random(m.shape) for m in masks_list]
    cp_measure.set_accelerator("numba")
    try:
        feret = cp_measure.bulk.get_core_measurements()["feret"]
        got = feret(masks_list, pixels_list)
    finally:
        cp_measure.set_accelerator(None)
    assert isinstance(got, list) and len(got) == 2
    for m, p, g in zip(masks_list, pixels_list, got):
        _assert_match(feret_numpy(m, p), g)
