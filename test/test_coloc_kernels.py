"""Unit tests for the colocalization numba primitives."""

import numpy as np
import pytest
import scipy.stats

from cp_measure._detect import HAS_NUMBA

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


@requires_numba
def test_flatten_pairs_grouped_blocks():
    from cp_measure.primitives._segment_numba import flatten_pairs_grouped
    from cp_measure.primitives.segment import labels_to_offsets

    masks = np.array([[0, 1, 1], [2, 2, 0], [2, 0, 1]], np.int64)
    p1 = np.arange(9, dtype=np.float64).reshape(3, 3)
    p2 = (np.arange(9, dtype=np.float64) * 10).reshape(3, 3)
    lut, n, offsets = labels_to_offsets(masks[np.newaxis])
    g1, g2 = flatten_pairs_grouped(
        masks[np.newaxis], p1[np.newaxis], p2[np.newaxis], lut, offsets
    )

    assert n == 2
    assert list(offsets) == [0, 3, 6]  # label 1 has 3 px, label 2 has 3 px
    # Each object's block is exactly the (channel-aligned) masked pixels.
    assert sorted(g1[0:3]) == sorted(p1[masks == 1])
    assert sorted(g2[3:6]) == sorted(p2[masks == 2])
    # g1/g2 stay co-registered: g2 == g1 * 10 elementwise everywhere.
    np.testing.assert_array_equal(g2, g1 * 10)


@requires_numba
def test_flatten_pairs_grouped_keeps_nonfinite():
    """Reference extracts pixels[mask] with no finiteness filter — match it."""
    from cp_measure.primitives._segment_numba import flatten_pairs_grouped
    from cp_measure.primitives.segment import labels_to_offsets

    masks = np.array([[1, 1, 1]], np.int64)
    p1 = np.array([[1.0, np.nan, 3.0]])
    p2 = np.array([[1.0, 2.0, np.inf]])
    lut, n, offsets = labels_to_offsets(masks[np.newaxis])
    g1, g2 = flatten_pairs_grouped(
        masks[np.newaxis], p1[np.newaxis], p2[np.newaxis], lut, offsets
    )
    assert offsets[-1] == 3  # all three pixels kept
    assert np.isnan(g1).sum() == 1 and np.isinf(g2).sum() == 1


@pytest.mark.parametrize("labels", [[0, 1, 2, 3], [0, 2, 5, 5, 2], [0, 0, 0]])
def test_labels_to_offsets_agrees_with_lut(labels):
    """bincount-based (lut, n) == find_objects-based; offsets are the CSR counts."""
    from cp_measure.primitives.segment import label_to_idx_lut, labels_to_offsets

    masks = np.array(labels, np.int64).reshape(1, 1, -1)
    lut_ref, n_ref = label_to_idx_lut(masks)
    lut, n, offsets = labels_to_offsets(masks)
    assert n == n_ref
    np.testing.assert_array_equal(lut, lut_ref)
    assert offsets[0] == 0 and offsets[-1] == int((masks > 0).sum())
    np.testing.assert_array_equal(np.diff(offsets), np.bincount(lut[masks[masks > 0]]))


@requires_numba
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dense_rank_matches_scipy(seed):
    """0-based dense rank == scipy.stats.rankdata(method='dense') - 1, with ties."""
    from cp_measure.core.numba._colocalization import _dense_rank

    rng = np.random.default_rng(seed)
    # Small integer range forces ties.
    vals = rng.integers(0, 5, size=40).astype(np.float64)
    ranks, rmax = _dense_rank(vals)
    expected = scipy.stats.rankdata(vals, method="dense") - 1
    np.testing.assert_array_equal(ranks, expected)
    assert rmax == expected.max()
