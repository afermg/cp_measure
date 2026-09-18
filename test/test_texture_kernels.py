"""Unit tests for the numba Haralick texture kernel (vs mahotas)."""

import mahotas.features
import numpy as np
import pytest

from cp_measure._detect import HAS_NUMBA

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")

# Variance and Correlation use the centred form where mahotas uses the raw
# dot(px, k**2) - ux**2, which loses up to ~1e-6 relative to cancellation on
# low-contrast objects; the rest agree far more tightly. See ``_texture``.
_LOOSE = {2: 1e-5, 3: 1e-5}


def _assert_features_close(got, exp):
    """Per-feature comparison against mahotas, loosened only where we differ."""
    for index in range(13):
        np.testing.assert_allclose(
            got[:, index],
            exp[:, index],
            rtol=_LOOSE.get(index, 1e-6),
            atol=1e-8,
            equal_nan=True,
            err_msg=f"feature {index}",
        )


def _kernel(crop, scale, deltas):
    from cp_measure.core.numba._texture import haralick_object

    crop = np.ascontiguousarray(crop)
    if crop.ndim == 2:
        crop = crop[np.newaxis]
    return haralick_object(crop, np.ascontiguousarray(scale * deltas))


@requires_numba
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("scale", [1, 3])
def test_haralick_2d_matches_mahotas(seed, scale):
    from cp_measure.core.numba._texture import DELTAS_2D

    rng = np.random.default_rng(seed)
    crop = rng.integers(0, 16, (24, 20)).astype(np.uint8)  # 0s exercise ignore_zeros
    _assert_features_close(
        _kernel(crop, scale, DELTAS_2D),
        mahotas.features.haralick(crop, distance=scale, ignore_zeros=True),
    )


@requires_numba
def test_haralick_3d_matches_mahotas():
    from cp_measure.core.numba._texture import DELTAS_3D

    rng = np.random.default_rng(0)
    crop = rng.integers(0, 16, (8, 12, 10)).astype(np.uint8)
    _assert_features_close(
        _kernel(crop, 3, DELTAS_3D),
        mahotas.features.haralick(crop, distance=3, ignore_zeros=True),
    )


@requires_numba
def test_empty_glcm_all_nan():
    """All-background crop -> empty GLCM after ignore_zeros -> all NaN (mahotas raises)."""
    from cp_measure.core.numba._texture import DELTAS_2D

    crop = np.zeros((10, 10), np.uint8)
    assert np.all(np.isnan(_kernel(crop, 3, DELTAS_2D)))
    with pytest.raises(ValueError):
        mahotas.features.haralick(crop, distance=3, ignore_zeros=True)


@requires_numba
@pytest.mark.parametrize(
    "size, scale", [(12, 3), (8, 1), (10, 2)], ids=["T=216", "T=98", "T=162"]
)
def test_constant_crop_matches_mahotas(size, scale):
    """Uniform object -> zero variance -> Correlation is 1 and InfoMeas1 is 0.

    The pair count T decides whether ``1.0 / T`` round-trips exactly, so the
    degeneracy is only reachable for some (size, scale): at T=98 a reciprocal
    leaves a residual variance, the ``sx == 0`` guard misses and Correlation
    comes back ~1e15. Dividing by T instead keeps all three cases exact.
    """
    from cp_measure.core.numba._texture import DELTAS_2D

    crop = np.full((size, size), 5, np.uint8)
    got = _kernel(crop, scale, DELTAS_2D)
    exp = mahotas.features.haralick(crop, distance=scale, ignore_zeros=True)
    np.testing.assert_array_equal(got[:, 2], 1.0)  # Correlation
    np.testing.assert_array_equal(got[:, 11], 0.0)  # InfoMeas1
    _assert_features_close(got, exp)
