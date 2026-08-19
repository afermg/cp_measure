"""Unit tests for the numba Haralick texture kernel (vs mahotas)."""

import mahotas.features
import numpy as np
import pytest

from cp_measure._detect import HAS_NUMBA

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


@requires_numba
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("scale", [1, 3])
def test_haralick_2d_matches_mahotas(seed, scale):
    from cp_measure.core.numba._texture import DELTAS_2D, haralick_object

    rng = np.random.default_rng(seed)
    crop = rng.integers(0, 16, (24, 20)).astype(np.uint8)  # 0s exercise ignore_zeros
    got = haralick_object(
        np.ascontiguousarray(crop.astype(np.int64))[np.newaxis],
        np.ascontiguousarray(scale * DELTAS_2D),
    )
    exp = mahotas.features.haralick(crop, distance=scale, ignore_zeros=True)
    np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-8)


@requires_numba
def test_haralick_3d_matches_mahotas():
    from cp_measure.core.numba._texture import DELTAS_3D, haralick_object

    rng = np.random.default_rng(0)
    crop = rng.integers(0, 16, (8, 12, 10)).astype(np.uint8)
    got = haralick_object(
        np.ascontiguousarray(crop.astype(np.int64)),
        np.ascontiguousarray(3 * DELTAS_3D),
    )
    exp = mahotas.features.haralick(crop, distance=3, ignore_zeros=True)
    np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-8)


@requires_numba
def test_empty_glcm_all_nan():
    """All-background crop -> empty GLCM after ignore_zeros -> all NaN (mahotas raises)."""
    from cp_measure.core.numba._texture import DELTAS_2D, haralick_object

    crop = np.zeros((10, 10), np.int64)
    got = haralick_object(crop[np.newaxis], np.ascontiguousarray(3 * DELTAS_2D))
    assert np.all(np.isnan(got))
    with pytest.raises(ValueError):
        mahotas.features.haralick(crop.astype(np.uint8), distance=3, ignore_zeros=True)


@requires_numba
def test_constant_crop_matches_mahotas():
    """Constant non-zero crop -> sx==0 -> Correlation==1; match mahotas elsewhere."""
    from cp_measure.core.numba._texture import DELTAS_2D, haralick_object

    crop = np.full((12, 12), 5, np.int64)
    got = haralick_object(crop[np.newaxis], np.ascontiguousarray(3 * DELTAS_2D))
    exp = mahotas.features.haralick(
        crop.astype(np.uint8), distance=3, ignore_zeros=True
    )
    np.testing.assert_allclose(got[:, 2], 1.0)  # Correlation
    np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-8, equal_nan=True)
