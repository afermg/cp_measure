"""Backend correctness: numba colocalization must match the numpy backend.

Compares the numba ``pearson``/``manders_fold``/``rwc``/``overlap`` against the
reference ``cp_measure.core.measurecolocalization`` key-by-key. Two value regimes
run: continuous random floats (generic path) and integer-VALUED floats in a small
range (forces RWC dense-rank ties). 2D and 3D, single image and batch (4D array +
ragged list) all run, since every bzyx path must hold.

Pixels stay float64 deliberately. Feeding a genuine integer dtype to the *numpy*
reference triggers two baseline artifacts the numba backend does not reproduce
(it upcasts to float64): ``fi*si`` overflows in uint8 (so Overlap/K can exceed 1),
and the Pearson slope's ``lstsq`` design matrix is uint8 → a float32 result. The
numba float64 path is strictly more accurate; matching those artifacts is not a
goal. (costes, the follow-up, will exercise true integer dtypes for its scale.)
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

FUNCS = ["pearson", "manders_fold", "rwc", "overlap", "costes"]


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


@requires_numba
@pytest.mark.parametrize("name", FUNCS)
def test_batch_list_matches_per_image(name):
    imgs = [_data_2d(float), _data_3d(float)]
    masks = [m for m, _, _ in imgs]
    p1 = [a for _, a, _ in imgs]
    p2 = [b for _, _, b in imgs]
    got = _numba_fn(name)(p1, p2, masks)
    assert isinstance(got, list) and len(got) == 2
    for (m, a, b), per_image in zip(imgs, got):
        _assert_match(per_image, getattr(ref, f"get_correlation_{name}")(a, b, m))


@requires_numba
@pytest.mark.parametrize("name", FUNCS)
def test_4d_batch_matches_per_image(name):
    m0, a0, b0 = _data_2d("cont")
    m1, a1, b1 = _data_2d("ties")
    masks = np.stack([m0[np.newaxis], m1[np.newaxis]])  # (2, 1, H, W)
    p1 = np.stack([a0[np.newaxis], a1[np.newaxis]])
    p2 = np.stack([b0[np.newaxis], b1[np.newaxis]])
    got = _numba_fn(name)(p1, p2, masks)
    assert isinstance(got, list) and len(got) == 2
    for per_image, (m, a, b) in zip(got, [(m0, a0, b0), (m1, a1, b1)]):
        _assert_match(per_image, getattr(ref, f"get_correlation_{name}")(a, b, m))


@requires_numba
@pytest.mark.parametrize("mode", ["Faster", "Fast", "Accurate"])
@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_costes_modes_match_numpy(mode, dim):
    masks, p1, p2 = _data_2d("cont") if dim == "2d" else _data_3d("cont")
    expected = ref.get_correlation_costes(p1, p2, masks, fast_costes=mode)
    got = _numba_fn("costes")(p1, p2, masks, fast_costes=mode)
    _assert_match(got, expected)
