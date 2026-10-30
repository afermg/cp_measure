"""Current-main contracts for the non-Costes colocalization accelerator."""

import inspect
import warnings

import numpy as np
import pytest

import cp_measure
from cp_measure import bulk
from cp_measure._detect import HAS_NUMBA
from cp_measure._sanitize import sanitize_masks
from cp_measure.core import measurecolocalization as ref

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
NAMES = ("pearson", "manders_fold", "rwc", "overlap")


def functions(name):
    from cp_measure.core.numba import measurecolocalization as nb

    return tuple(getattr(mod, f"get_correlation_{name}") for mod in (ref, nb))


def assert_parity(name, first, second, masks, **kwargs):
    reference, accelerated = functions(name)
    originals = [x.copy() for x in (first, second, masks)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            expected = reference(first, second, masks, **kwargs)
        except ZeroDivisionError:
            # Current-main overlap uses Python float division for K1/K2.
            with pytest.raises(ZeroDivisionError):
                accelerated(first, second, masks, **kwargs)
            return
        actual = accelerated(first, second, masks, **kwargs)
    assert list(actual) == list(expected)
    for key in expected:
        assert isinstance(actual[key], type(expected[key]))
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
        assert np.asarray(actual[key]).dtype == np.asarray(expected[key]).dtype
    for original, value in zip(originals, (first, second, masks)):
        np.testing.assert_array_equal(original, value)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "case",
    [
        "zero",
        "constant",
        "one_pixel",
        "nan",
        "inf",
        "gaps",
        "empty",
        "negative",
        "no_combined",
    ],
)
def test_current_main_undefined_and_missing_labels(name, case):
    first = np.array([[0.2, 0.8, 0.4], [0.6, 1.0, 0.3]])
    second = first[:, ::-1].copy()
    masks = np.ones(first.shape, dtype=np.int32)
    if case == "zero":
        first[:] = 0
    elif case == "constant":
        first[:] = 2
    elif case == "one_pixel":
        masks[:] = 0
        masks[0, 0] = 1
    elif case == "nan":
        first[0, 0] = np.nan
    elif case == "inf":
        first[0, 0] = np.inf
    elif case == "gaps":
        masks[1] = 3
    elif case == "empty":
        masks[:] = 0
    elif case == "negative":
        first *= -1
    elif case == "no_combined":
        first[:] = [[1, 0, 0], [0, 0, 0]]
        second[:] = [[0, 1, 0], [0, 0, 0]]
    assert_parity(name, first, second, masks)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64]
)
@pytest.mark.parametrize("thr", [0, 15, 73.5])
def test_current_main_dtype_and_threshold(name, dtype, thr):
    rng = np.random.default_rng(60)
    first = rng.integers(1, 250, size=(7, 8)).astype(dtype)
    second = rng.integers(1, 250, size=(7, 8)).astype(dtype)
    masks = np.ones(first.shape, np.int32)
    masks[3:] = 2
    kwargs = {} if name == "pearson" else {"thr": thr}
    assert_parity(name, first, second, masks, **kwargs)


@pytest.mark.parametrize("name", NAMES)
def test_float32_threshold_boundary_keeps_current_numpy_semantics(name):
    # Do not implicitly bundle #124's threshold promotion fix into this backend.
    first = np.array([[3, 20], [10, 0]], dtype=np.float32) / 65535
    second = np.array([[20, 3], [10, 0]], dtype=np.float32) / 65535
    assert_parity(name, first, second, np.ones(first.shape, np.int32))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "shapes",
    [
        ((3, 4), (1, 3, 4), (3, 4)),
        ((3, 4), (3, 4), (2, 3, 4)),
        ((3, 4), (3, 5), (3, 4)),
        ((3, 4), (3, 4), (3, 5)),
    ],
)
def test_rejects_mismatched_spatial_shapes_before_kernel(name, shapes):
    first, second = [np.ones(s) for s in shapes[:2]]
    masks = np.ones(shapes[2], np.int32)
    with pytest.raises(ValueError, match="shape"):
        functions(name)[1](first, second, masks)


@pytest.mark.parametrize("name", NAMES)
def test_signature_and_noncontiguous_singleton_volume(name):
    reference, accelerated = functions(name)
    ref_params = inspect.signature(reference).parameters
    nb_params = inspect.signature(accelerated).parameters
    assert list(ref_params) == list(nb_params)
    for key in ref_params:
        assert ref_params[key].default == nb_params[key].default
        assert ref_params[key].kind == nb_params[key].kind
    rng = np.random.default_rng(601)
    first, second = [rng.random((1, 8, 12))[:, ::2, ::-2] for _ in range(2)]
    masks = np.ones(first.shape, np.int32)
    assert_parity(name, first, second, masks)


@pytest.mark.parametrize("name", ["pearson", "manders_fold", "rwc"])
def test_bulk_label_mapping_and_keyword_forwarding(name):
    rng = np.random.default_rng(602)
    masks = np.array([[0, 42, 42, 42], [0, 7, 7, 7]], dtype=np.int64)
    first, second = [rng.random(masks.shape) for _ in range(2)]
    clean, ids = sanitize_masks(masks)
    assert list(ids) == [7, 42]
    kwargs = {} if name == "pearson" else {"thr": 73.5}
    expected = functions(name)[0](first, second, clean, **kwargs)
    try:
        cp_measure.set_accelerator("numba")
        actual = bulk.get_correlation_measurements()[name](
            pixels_1=first, pixels_2=second, masks=masks, **kwargs
        )
    finally:
        cp_measure.set_accelerator(None)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)


def test_rwc_default_is_object_local():
    first = np.array([[0.2, 0.4, 0.8], [0.3, 0.6, 0.9]])
    second = np.array([[0.8, 0.2, 0.4], [0.7, 0.5, 0.3]])
    masks = np.array([[1, 1, 1], [2, 2, 2]], np.int32)
    accelerated = functions("rwc")[1]
    together = accelerated(first, second, masks)
    alone = accelerated(first[:1], second[:1], masks[:1])
    for key in alone:
        assert together[key][0] == alone[key][0]
    assert_parity("rwc", first, second, masks)


@pytest.mark.parametrize("is_3d", [False, True])
def test_featurizer_colocalization_schema_rows_and_forwarding(is_3d):
    from conftest import ALL_OFF
    from cp_measure.featurizer import featurize, make_featurizer_config

    rng = np.random.default_rng(603)
    images = rng.random((2, 2, 6, 8))
    masks = np.zeros((2, 1, 6, 8), np.int32)
    masks[0, 0, :3] = 42
    masks[0, 0, 3:] = 7
    masks[1, 0, :3] = 10
    masks[1, 0, 3:] = 4
    if is_3d:
        images = images[:, :, None]
        masks = masks[:, :, None]
    config = make_featurizer_config(
        ["first", "second"],
        **{
            **ALL_OFF,
            "correlation_pearson": True,
            "correlation_manders_fold": True,
            "correlation_rwc": True,
        },
        correlation_manders_fold_params={"thr": 73.5},
        correlation_rwc_params={"thr": 61},
    )
    try:
        cp_measure.set_accelerator(None)
        expected, columns, rows = featurize(images, masks, config, is_3d=is_3d)
        cp_measure.set_accelerator("numba")
        actual, nb_columns, nb_rows = featurize(images, masks, config, is_3d=is_3d)
    finally:
        cp_measure.set_accelerator(None)
    assert nb_columns == columns
    assert nb_rows == rows
    assert [row[-1] for row in rows] == [7, 42, 4, 10]
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("count", [8, 10, 16, 100, 128, 1024])
def test_pearson_constant_nonbinary_mean_keeps_nan_contract(count):
    first = np.full((1, count), 0.1)
    second = np.arange(count, dtype=float).reshape(1, count)
    assert_parity("pearson", first, second, np.ones(first.shape, np.int32))


def test_pearson_large_offset_reduction_parity():
    rng = np.random.default_rng(6)
    first, second = [1e12 + rng.random((1, 1000)) for _ in range(2)]
    assert_parity("pearson", first, second, np.ones(first.shape, np.int32))


@pytest.mark.parametrize("name", NAMES)
def test_uint64_label_slots(name):
    first = np.array([[0.1, 0.2, 0.5], [0.3, 0.4, 0.6]])
    second = first[:, ::-1].copy()
    masks = np.array([[1, 1, 1], [3, 3, 3]], dtype=np.uint64)
    assert_parity(name, first, second, masks)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("label_type", ["i2", "u2", "i4", "u4", "i8", "u8"])
@pytest.mark.parametrize("byteorder", ["=", "S"], ids=["native", "swapped"])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_label_byteorder_parity_and_immutability(name, label_type, byteorder, strided):
    dtype = np.dtype(label_type).newbyteorder(byteorder)
    masks = np.ones((3, 8 if strided else 4), dtype=dtype)
    if strided:
        masks = masks[:, ::2]
        masks[-1] = 3  # Retain a missing raw label slot too.
    masks.setflags(write=False)
    original_bytes = masks.tobytes()
    first = np.arange(1, 13, dtype=float).reshape(3, 4)
    second = first[::-1].copy()

    assert_parity(name, first, second, masks)

    assert masks.dtype == dtype
    assert masks.tobytes() == original_bytes
    assert not masks.flags.writeable
