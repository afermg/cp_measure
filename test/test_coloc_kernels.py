"""Unit tests for the colocalization numba primitives."""

import numpy as np
import pytest
import scipy.stats

from cp_measure._detect import HAS_NUMBA

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


def _offsets(masks):
    """CSR block bounds for contiguous labels 1..N — the cumulative pixel counts."""
    n = int(masks.max())
    offsets = np.zeros(n + 1, np.int64)
    offsets[1:] = np.cumsum(np.bincount(masks.ravel(), minlength=n + 1)[1:])
    return offsets


@requires_numba
def test_flatten_pairs_grouped_blocks():
    from cp_measure.primitives._segment_numba import flatten_pairs_grouped

    masks = np.array([[0, 1, 1], [2, 2, 0], [2, 0, 1]], np.int64)
    p1 = np.arange(9, dtype=np.float64).reshape(3, 3)
    p2 = (np.arange(9, dtype=np.float64) * 10).reshape(3, 3)
    offsets = _offsets(masks[np.newaxis])
    g1, g2 = flatten_pairs_grouped(
        masks[np.newaxis], p1[np.newaxis], p2[np.newaxis], offsets
    )

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

    masks = np.array([[1, 1, 1]], np.int64)
    p1 = np.array([[1.0, np.nan, 3.0]])
    p2 = np.array([[1.0, 2.0, np.inf]])
    offsets = _offsets(masks[np.newaxis])
    g1, g2 = flatten_pairs_grouped(
        masks[np.newaxis], p1[np.newaxis], p2[np.newaxis], offsets
    )
    assert offsets[-1] == 3  # all three pixels kept
    assert np.isnan(g1).sum() == 1 and np.isinf(g2).sum() == 1


@requires_numba
@pytest.mark.parametrize("label_dtype", [np.int32, np.uint16, np.uint64])
@pytest.mark.parametrize("pixel_dtype", [np.float32, np.float64])
def test_group_pairs_matches_host_grouping(label_dtype, pixel_dtype):
    """One kernel call finds the labels, the block bounds and both channels' blocks,
    reading float32 pixels directly (widened exactly), as the host steps did."""
    from cp_measure.primitives._segment_numba import (
        flatten_pairs_grouped,
        group_pairs,
    )

    masks = np.array([[[0, 1, 1], [4, 4, 0], [4, 0, 1]]], label_dtype)  # 2, 3 missing
    p1 = np.arange(9, dtype=pixel_dtype).reshape(1, 3, 3) / 7
    p2 = p1[:, ::-1].copy()
    p1[0, 2, 2] = np.nan
    g1, g2, offsets = group_pairs(masks, p1, p2)
    expected_offsets = _offsets(masks.astype(np.int64))
    np.testing.assert_array_equal(offsets, expected_offsets)
    expected = flatten_pairs_grouped(
        masks, p1.astype(np.float64), p2.astype(np.float64), expected_offsets
    )
    for got, want in zip((g1, g2), expected):
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, want)


@requires_numba
def test_group_pairs_without_objects():
    from cp_measure.primitives._segment_numba import group_pairs

    g1, g2, offsets = group_pairs(np.zeros((1, 2, 2), np.int32), *np.ones((2, 1, 2, 2)))
    assert list(offsets) == [0] and g1.size == g2.size == 0


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


@requires_numba
def test_fused_kernel_undefined_and_missing_slots():
    from cp_measure.core.numba._colocalization import coloc_per_object

    first = np.array([0.0, 0.0, 0.0, 1.0, np.nan, 2.0])
    second = np.array([0.0, 0.0, 0.0, 2.0, 1.0, 3.0])
    result = coloc_per_object(first, second, np.array([0, 3, 3, 6]), 3, 0.15, True)
    for values in result:
        assert values[1] == 0  # missing label retains its slot
        assert np.isnan(values[0])  # zero denominators, not ZeroDivisionError
    assert np.isnan(result[0][2])
    assert np.isnan(result[1][2])
    for values in result[2:]:
        assert values[2] == 0  # NaN max means no combined-threshold pixels


@requires_numba
@pytest.mark.parametrize(
    "n", [1, 2, 7, 8, 9, 127, 128, 129, 1000, 8191, 8192, 8193, 16385, 100_003]
)
@pytest.mark.parametrize("bufsize", [None, 4096])
def test_numpy_sum_and_var_match_numpy_bitwise(n, bufsize):
    """The JIT ports reproduce NumPy's float64 sum, mean and ddof=1 variance exactly,
    including the pre-2.3 chunking by the current buffer size and signed zeros."""
    previous = np.getbufsize()
    if bufsize:
        np.setbufsize(bufsize)
    try:
        _check_numpy_sum_and_var(n)
    finally:
        np.setbufsize(previous)


def test_numpy_var_singleton_is_nan():
    """ddof=1 variance of a singleton is 0/0 = NaN (as numpy.var), not a crash:
    numpy_var carries error_model='numpy' like the other segment kernels, so the
    degenerate denominator returns NaN instead of raising ZeroDivisionError."""
    from cp_measure.primitives._segment_numba import numpy_sum_chunk, numpy_var

    assert np.isnan(numpy_var(np.array([3.0]), numpy_sum_chunk()))


def _check_numpy_sum_and_var(n):
    from cp_measure.primitives._segment_numba import (
        numpy_sum,
        numpy_sum_chunk,
        numpy_var,
    )

    zeros = np.full(n, -0.0)
    assert np.signbit(numpy_sum(zeros, numpy_sum_chunk())) == np.signbit(np.sum(zeros))
    rng = np.random.default_rng(n)
    for scale, offset in [(1.0, 0.0), (1e-3, 0.5), (1e3, -7.0)]:
        values = rng.random(n) * scale + offset
        chunk = numpy_sum_chunk()
        assert numpy_sum(values, chunk) == np.sum(values)
        assert numpy_sum(values, chunk) / n == np.mean(values)
        if n > 1:
            assert numpy_var(values, chunk) == np.var(values, ddof=1)
