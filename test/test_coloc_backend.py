"""Backend correctness: numba colocalization must match the numpy backend.

Compares the numba ``pearson``/``manders_fold``/``rwc``/``overlap`` against the
reference ``cp_measure.core.measurecolocalization`` key-by-key. Two value regimes
run: continuous random floats (generic path) and integer-VALUED floats in a small
range (forces RWC dense-rank ties), in 2D and 3D. Shared batch/container,
ordering and immutability checks live in test_batch_contract.py.

These golden inputs exercise the accelerated float64 path. Dtype-sensitive
reference compatibility (including real integer images) is covered separately
in test_coloc_contracts.py.
"""

import numpy as np
import pytest
from conftest import (
    DEPTH_3D,
    SIZE_2D,
    SIZE_3D,
    _stamp_objects_2d,
    _stamp_objects_3d,
    get_rng,
)

import cp_measure.core.measurecolocalization as ref
from cp_measure._detect import HAS_NUMBA

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")

FUNCS = ["pearson", "manders_fold", "rwc", "overlap"]


def _numba_fn(name):
    import cp_measure.core.numba.measurecolocalization as nb

    return getattr(nb, f"get_correlation_{name}")


def _pixels(shape, regime):
    rng = get_rng()
    if regime == "cont":
        return rng.random(shape), rng.random(shape)
    # Integer-valued floats in a small range -> many RWC rank ties, no dtype artifact.
    return (
        rng.integers(0, 8, shape).astype(np.float64),
        rng.integers(0, 8, shape).astype(np.float64),
    )


def _data_2d(regime="cont"):
    masks = np.zeros((SIZE_2D, SIZE_2D), np.int32)
    _stamp_objects_2d(masks, n_objects=3)
    p1, p2 = _pixels((SIZE_2D, SIZE_2D), regime)
    return masks, p1, p2


def _data_3d(regime="cont"):
    masks = np.zeros((DEPTH_3D, SIZE_3D, SIZE_3D), np.int32)
    _stamp_objects_3d(masks, n_objects=2)
    p1, p2 = _pixels((DEPTH_3D, SIZE_3D, SIZE_3D), regime)
    return masks, p1, p2


def _assert_match(got, expected):
    assert set(got) == set(expected), set(got).symmetric_difference(expected)
    for key in expected:
        np.testing.assert_allclose(
            got[key], expected[key], rtol=1e-6, atol=1e-8, err_msg=f"feature {key!r}"
        )


@requires_numba
@pytest.mark.parametrize("name", FUNCS)
@pytest.mark.parametrize("regime", ["cont", "ties"])
def test_matches_numpy_2d(name, regime):
    masks, p1, p2 = _data_2d(regime)
    expected = getattr(ref, f"get_correlation_{name}")(p1, p2, masks)
    got = _numba_fn(name)(p1, p2, masks)
    _assert_match(got, expected)


@requires_numba
@pytest.mark.parametrize("name", FUNCS)
def test_matches_numpy_3d(name):
    masks, p1, p2 = _data_3d(float)
    expected = getattr(ref, f"get_correlation_{name}")(p1, p2, masks)
    got = _numba_fn(name)(p1, p2, masks)
    _assert_match(got, expected)
