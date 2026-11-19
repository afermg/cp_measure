"""Equivalence and dispatch tests for NumPy and Numba Feret measurements.

``feret_numpy`` is the raw ``core.measureobjectsizeshape.get_feret`` function,
not the featurizer or the sanitized bulk wrapper. Tests distinguish raw backend
math, Numba list-batch dispatch, and the public bulk wrapper's label cleanup.
Feret ignores pixel values and supports only 2D masks. Both raw backends retain
the ``{}`` result for multi-slice volumes and 2D results for singleton-Z inputs.
"""

import numpy as np
import pytest
from conftest import get_rng

import cp_measure
from cp_measure._detect import HAS_NUMBA
from cp_measure._sanitize import sanitize_masks
from cp_measure.bulk import get_core_measurements
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
    """The featurizer passes None for pixels to shape-only measurements."""
    from cp_measure.core.numba._feret import get_feret as feret_nb

    masks = _rich_mask()
    _assert_match(feret_numpy(masks, None), feret_nb(masks, None))


@requires_numba
def test_feret_bool_mask_matches_numpy():
    """Raw NumPy Feret accepts a bool mask; its Numba replacement must too.

    This tests the raw functions, not the bulk wrapper, which requires integers.
    """
    from cp_measure.core.numba._feret import get_feret as feret_nb

    m = np.zeros((20, 20), dtype=bool)
    m[3:15, 4:12] = True
    pixels = get_rng().random(m.shape)
    _assert_match(feret_numpy(m, pixels), feret_nb(m, pixels))


@requires_numba
def test_feret_3d_returns_empty():
    """Preserve raw NumPy behavior; the featurizer filters 2D-only features earlier."""
    from cp_measure.core.numba._feret import get_feret as feret_nb

    masks = np.zeros((4, 16, 16), np.int32)
    masks[1:3, 4:10, 4:10] = 1
    pixels = get_rng().random(masks.shape)
    assert feret_nb(masks, pixels) == {}
    assert feret_numpy(masks, pixels) == {}


@requires_numba
def test_feret_empty_mask_matches_numpy_raises():
    """Raw NumPy Feret raises on an empty convex hull; Numba must match.

    Empty-image skipping belongs to the featurizer, not this raw-function test.
    """
    from cp_measure.core.numba._feret import get_feret as feret_nb

    masks = np.zeros((16, 16), np.int32)
    pixels = get_rng().random(masks.shape)
    with pytest.raises(ValueError):
        feret_numpy(masks, pixels)
    with pytest.raises(ValueError):
        feret_nb(masks, pixels)


@requires_numba
def test_feret_single_slice_volume_uses_2d_measurement():
    """A singleton-Z volume is eligible for the 2D Feret measurement."""
    from cp_measure.core.numba._feret import get_feret as feret_nb

    masks = _rich_mask()[np.newaxis]  # (1, 80, 80)
    pixels = get_rng().random(masks.shape)
    expected = feret_numpy(masks[0], pixels[0])
    _assert_match(feret_nb(masks, pixels), expected)
    _assert_match(feret_numpy(masks, pixels), expected)


@requires_numba
def test_feret_non_native_byteorder_matches_numpy():
    """Numba rejects non-native byte order, although raw NumPy Feret accepts it.

    Conversion stays at the Numba boundary so bulk and raw callers benefit too.
    """
    from cp_measure.core.numba._feret import get_feret as feret_nb

    masks = _rich_mask().astype(np.dtype(np.int32).newbyteorder("S"))
    pixels = get_rng().random(masks.shape)
    _assert_match(feret_numpy(masks, pixels), feret_nb(masks, pixels))


@requires_numba
def test_feret_through_bulk_wrapper_sanitizes_each_image():
    """The bulk wrapper must relabel each mask separately before calling Numba.

    Images have different object counts and disjoint, gapped label IDs. Relabeling
    the whole batch at once would leave gaps in each image and produce extra rows.
    This tests get_core_measurements(), not featurize(). Inputs must stay unchanged.
    """
    first = _rich_mask() * 10
    second = _rich_mask()
    second[second > 3] = 0
    second = np.where(second > 0, second * 100 + 7, 0).astype(np.int32)
    masks_list = [first, second]
    originals = [m.copy() for m in masks_list]
    pixels_list = [get_rng().random(m.shape) for m in masks_list]
    previous_accelerator = cp_measure._ACCELERATOR
    cp_measure.set_accelerator("numba")
    try:
        feret = get_core_measurements()["feret"]
        got = feret(masks_list, pixels_list)
        individual = [feret(m, p) for m, p in zip(masks_list, pixels_list)]
    finally:
        cp_measure.set_accelerator(previous_accelerator)
    assert isinstance(got, list) and len(got) == 2
    for m, p, g, single, original in zip(
        masks_list, pixels_list, got, individual, originals
    ):
        clean, labels = sanitize_masks(m)
        _assert_match(feret_numpy(clean, p), g)
        _assert_match(single, g)
        assert all(values.shape == (len(labels),) for values in g.values())
        np.testing.assert_array_equal(m, original)


@requires_numba
@pytest.mark.parametrize("depth", [1, 3])
def test_feret_4d_batch_uses_each_volumes_z_depth(depth):
    """A batch uses the same sole-slice eligibility rule as individual volumes."""
    from cp_measure.core.numba._feret import get_feret as feret_nb

    masks = np.repeat(_rich_mask()[None, None], 2, axis=0)
    masks = np.repeat(masks, depth, axis=1)
    got = feret_nb(masks, None)
    assert len(got) == 2
    for result, mask in zip(got, masks):
        if depth == 1:
            _assert_match(result, feret_numpy(mask[0], None))
        else:
            assert result == {}
