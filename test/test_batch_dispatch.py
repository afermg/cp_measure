"""Contracts of the serial per-image decorator and its NumPy entry points."""

from inspect import signature, unwrap

import numpy as np
import pytest

from cp_measure.core.measureobjectintensity import get_intensity
from cp_measure.core.measuretexture import get_texture
from cp_measure.primitives.shapes import batch_dispatch


@pytest.mark.parametrize("measurement", [get_intensity, get_texture])
def test_metadata_and_signature(measurement):
    original = unwrap(measurement)
    assert original is measurement.__wrapped__
    assert signature(measurement) == signature(original)
    for attribute in (
        "__name__",
        "__qualname__",
        "__module__",
        "__doc__",
        "__annotations__",
    ):
        assert getattr(measurement, attribute) == getattr(original, attribute)


def test_arguments_and_shape_only_inputs():
    @batch_dispatch
    def measure(masks, pixels, scale=3, *extra, required, **options):
        return masks.ndim, pixels, scale, extra, required, options

    masks = np.ones((4, 5), np.int32)
    expected = (2, None, 7, ("extra",), "value", {"custom": 9})
    assert measure(masks, None, 7, "extra", required="value", custom=9) == expected
    assert measure([masks], None, 7, "extra", required="value", custom=9) == [expected]
    assert measure(masks=[masks], pixels=None, required="value") == [
        (2, None, 3, (), "value", {})
    ]


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((), {"unknown": True, "required": True}),
        ((1, 2, 3), {"required": True}),
        ((1,), {"scale": 2, "required": True}),
        ((), {}),
    ],
)
def test_invalid_arguments_precede_batch_validation(args, kwargs):
    @batch_dispatch
    def measure(masks, pixels, scale=3, *, required):
        pytest.fail("invalid arguments reached the implementation")

    with pytest.raises(TypeError):
        measure([], [], *args, **kwargs)


def test_batch_validation_precedes_measurement():
    @batch_dispatch
    def measure(masks, pixels):
        pytest.fail("invalid batch reached the implementation")

    with pytest.raises(ValueError, match="at least one"):
        measure([], [])
    masks = [np.ones((4, 5)), np.ones((3, 5))]
    pixels = [np.ones((4, 5)), np.ones((4, 5))]
    with pytest.raises(ValueError, match="matching shapes and rank"):
        measure(masks, pixels)


@pytest.mark.parametrize(
    "measurement, options",
    [
        (get_intensity, {"edge_measurements": False, "legacy": True}),
        (get_texture, {"scale": 1, "gray_levels": 8}),
    ],
)
@pytest.mark.parametrize("shape", [(8, 9), (1, 8, 9), (3, 8, 9)])
def test_numpy_entry_points_match_per_image_implementation(measurement, options, shape):
    masks = [np.zeros(shape, np.int32) for _ in range(2)]
    masks[0][..., 1:7, 1:8] = 1
    pixels = [
        np.random.default_rng(42).random(shape).astype(dtype)
        for dtype in (np.float32, np.float64)
    ]
    for array in (*masks, *pixels):
        array.setflags(write=False)
    result = measurement(masks=masks, pixels=pixels, **options)
    assert isinstance(result, list) and len(result) == 2
    for got, mask, image in zip(result, masks, pixels):
        expected = measurement.__wrapped__(mask, image, **options)
        single = measurement(mask, image, **options)
        assert isinstance(single, dict)
        assert got.keys() == single.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(got[key], expected[key])
            np.testing.assert_array_equal(single[key], expected[key])
            assert got[key].dtype == expected[key].dtype
