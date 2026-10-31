"""Forward explicit RWC scope without changing the accelerated default."""

import numpy as np
import pytest

import cp_measure
from cp_measure import bulk
from cp_measure._detect import HAS_NUMBA
from cp_measure.core import measurecolocalization as ref
from test_rwc_rank_scope import scene

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scope", ["object", "object_set"])
@pytest.mark.parametrize("shape", ["2d", "singleton", "volume", "strided"])
def test_scope_parity_and_immutability(dtype, scope, shape):
    from cp_measure.core.numba.measurecolocalization import get_correlation_rwc

    a, b, labels = scene()
    a, b = a.astype(dtype), b.astype(dtype)
    if shape == "singleton":
        a, b, labels = a[None], b[None], labels[None]
    elif shape == "volume":
        a, b, labels = (np.stack([v, v]) for v in (a, b, labels))
    elif shape == "strided":
        a, b, labels = (v[:, ::-1] for v in (a, b, labels))
    originals = [v.copy() for v in (a, b, labels)]
    for v in (a, b, labels):
        v.setflags(write=False)
    expected = ref.get_correlation_rwc(a, b, labels, thr=42, rank_scope=scope)
    actual = get_correlation_rwc(a, b, labels, thr=42, rank_scope=scope)
    assert list(actual) == list(expected)
    for key in expected:
        assert isinstance(actual[key], list)
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    for before, after in zip(originals, (a, b, labels)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("as_list", [False, True])
def test_object_set_batches_rank_each_image_separately(as_list):
    from cp_measure.core.numba.measurecolocalization import get_correlation_rwc

    a, b, labels = scene()
    first = np.stack([a[None], (1 - a)[None]]).astype(np.float64)
    second = np.stack([b[None], (b * 0.3)[None]]).astype(np.float64)
    masks = np.stack([labels[None], labels[None]])
    if as_list:
        first, second, masks = list(first), list(second), list(masks)
    actual = get_correlation_rwc(first, second, masks, rank_scope="object_set")
    assert len(actual) == 2
    for got, x, y, m in zip(actual, first, second, masks):
        assert got == ref.get_correlation_rwc(x, y, m, rank_scope="object_set")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_object_set_bulk_and_featurizer_forwarding(dtype):
    from cp_measure.featurizer import featurize, make_featurizer_config
    from conftest import ALL_OFF

    a, b, labels = scene()
    a, b = a.astype(dtype), b.astype(dtype)
    config = make_featurizer_config(
        ["A", "B"],
        **{**ALL_OFF, "correlation_rwc": True},
        correlation_rwc_params={"rank_scope": "object_set"},
    )
    images, masks = np.stack([a, b])[None], labels[None, None]
    try:
        cp_measure.set_accelerator(None)
        expected = featurize(images, masks, config)
        cp_measure.set_accelerator("numba")
        raw = bulk.get_correlation_measurements(sanitize=False)["rwc"]
        assert "numba" in raw.__module__
        assert bulk.get_correlation_measurements()["rwc"](
            a, b, labels, rank_scope="object_set"
        ) == ref.get_correlation_rwc(a, b, labels, rank_scope="object_set")
        actual = featurize(images, masks, config)
        assert actual[1:] == expected[1:]
        np.testing.assert_array_equal(actual[0], expected[0])
    finally:
        cp_measure.set_accelerator(None)


def test_scope_validation_and_nonfinite_foreground():
    from cp_measure.core.numba.measurecolocalization import get_correlation_rwc

    a, b, labels = scene()
    with pytest.raises(ValueError, match="rank_scope"):
        get_correlation_rwc(a, b, labels, rank_scope="image")
    a[labels == 1] = np.nan
    with pytest.raises(ValueError, match="finite foreground"):
        get_correlation_rwc(a, b, labels, rank_scope="object_set")
    with pytest.raises(ValueError, match="same shape"):
        get_correlation_rwc(a, b, labels[None], rank_scope="object_set")
