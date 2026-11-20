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


@pytest.mark.parametrize("shape", [(8, 9), (1, 8, 9), (3, 8, 9)])
@pytest.mark.parametrize("container", [None, list, tuple, np.stack])
def test_rank_dtype_views_and_return_type(shape, container):
    def identity(masks, pixels):
        return {"masks": masks, "pixels": pixels}

    wrapped = batch_dispatch(identity)
    masks = [np.ones(shape, dtype)[..., ::-1] for dtype in (np.int16, np.int32)]
    pixels = [np.ones(shape, dtype)[..., ::-1] for dtype in (np.float32, np.float64)]
    if container is np.stack and len(shape) == 2:
        masks, pixels = (
            [array[None] for array in arrays] for arrays in (masks, pixels)
        )
    if container is None:
        inputs = masks[0], pixels[0]
        sources = [inputs]
    else:
        inputs = container(masks), container(pixels)
        if container is np.stack:
            for array in inputs:
                array.setflags(write=False)
        sources = list(zip(*inputs))
    for pair in sources:
        for array in pair:
            array.setflags(write=False)
    result = wrapped(masks=inputs[0], pixels=inputs[1])
    assert isinstance(result, dict if container is None else list)
    results = [result] if container is None else result
    assert len(results) == len(sources)
    for got, pair in zip(results, sources):
        for key, source in zip(("masks", "pixels"), pair):
            assert got[key].shape == source.shape
            assert got[key].dtype == source.dtype
            assert np.shares_memory(got[key], source)
            assert not got[key].flags.writeable
            np.testing.assert_array_equal(got[key], source)


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
        ((), {"unknown": True}),
        ((1, 2, 3), {}),
        ((1,), {"scale": 2}),
    ],
)
def test_invalid_arguments_precede_batch_validation(args, kwargs):
    @batch_dispatch
    def measure(masks, pixels, scale=3):
        pytest.fail("invalid arguments reached the implementation")

    with pytest.raises(TypeError):
        measure([], [], *args, **kwargs)


def test_missing_keyword_only_argument_precedes_batch_validation():
    @batch_dispatch
    def measure(masks, pixels, *, required):
        pytest.fail("missing argument reached the implementation")

    with pytest.raises(TypeError):
        measure([], [])


@pytest.mark.parametrize(
    "masks, pixels, match",
    [
        ([], [], "at least one"),
        (np.zeros((0, 1, 4, 5)), np.zeros((0, 1, 4, 5)), "at least one"),
        ([np.ones((4, 5))], [], "batch size mismatch"),
        ([np.ones((4, 5))], np.ones((4, 5)), "both be sequences"),
        (np.ones((1, 1, 4, 5)), np.ones((1, 4, 5)), "both be 4D"),
        ([np.ones((4, 5))], [np.ones((1, 4, 5))], "matching shapes and rank"),
        (
            [np.ones((4, 5)), np.ones((3, 5))],
            [np.ones((4, 5)), np.ones((3, 5))],
            "same shape",
        ),
        ([np.ones(5)], [np.ones(5)], "2D or 3D"),
    ],
)
def test_batch_validation_precedes_measurement(masks, pixels, match):
    @batch_dispatch
    def measure(masks, pixels):
        pytest.fail("invalid batch reached the implementation")

    with pytest.raises(ValueError, match=match):
        measure(masks, pixels)


@pytest.mark.parametrize(
    "measurement, args, kwargs",
    [
        (get_intensity, (), {}),
        (get_intensity, (False, True), {}),
        (get_intensity, (), {"edge_measurements": False, "legacy": True}),
        (get_texture, (), {}),
        (get_texture, (1, 8), {}),
        (get_texture, (), {"scale": 1, "gray_levels": 8}),
    ],
)
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_numpy_entry_points_match_per_image_implementation(
    measurement, args, kwargs, depth
):
    shape = (depth, 8, 9) if depth else (8, 9)
    masks = [np.zeros(shape, np.int32) for _ in range(2)]
    masks[0][..., 1:7, 1:8] = 1
    pixels = [
        np.random.default_rng(42).random(shape).astype(dtype)
        for dtype in (np.float32, np.float64)
    ]
    originals = [array.copy() for array in (*masks, *pixels)]
    for array in (*masks, *pixels):
        array.setflags(write=False)
    if args:
        result = measurement(masks, pixels, *args, **kwargs)
    else:
        result = measurement(masks=masks, pixels=pixels, **kwargs)
    assert isinstance(result, list) and len(result) == 2
    for got, mask, image in zip(result, masks, pixels):
        expected = measurement.__wrapped__(mask, image, *args, **kwargs)
        single = measurement(mask, image, *args, **kwargs)
        assert isinstance(single, dict)
        assert got.keys() == single.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(got[key], expected[key])
            np.testing.assert_array_equal(single[key], expected[key])
            assert got[key].dtype == expected[key].dtype
    for array, original in zip((*masks, *pixels), originals):
        np.testing.assert_array_equal(array, original)
