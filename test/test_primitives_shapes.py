"""Tests for the canonical (B,Z,Y,X) input normalisation helper."""

import numpy
import pytest

from cp_measure.primitives.shapes import each_volume, to_bzyx


def _img(shape, seed):
    rng = numpy.random.default_rng(seed)
    return rng.random(shape)


def test_2d_is_single_unit_z():
    m, p = numpy.ones((4, 5), int), _img((4, 5), 0)
    masks, pixels, unwrap = to_bzyx(m, p)
    assert len(masks) == len(pixels) == 1
    assert masks[0].shape == (1, 4, 5)
    assert pixels[0].shape == (1, 4, 5)
    # single input -> unwrap returns the lone result, not a list
    assert unwrap(["only"]) == "only"


def test_3d_is_single_volume_not_batch():
    m, p = numpy.ones((3, 4, 5), int), _img((3, 4, 5), 1)
    masks, pixels, unwrap = to_bzyx(m, p)
    assert len(masks) == 1
    assert masks[0].shape == (3, 4, 5)  # one volume, Z preserved
    assert unwrap(["d"]) == "d"


def test_4d_is_batch():
    m, p = numpy.ones((2, 3, 4, 5), int), _img((2, 3, 4, 5), 2)
    masks, pixels, unwrap = to_bzyx(m, p)
    assert isinstance(masks, numpy.ndarray) and masks.shape == (2, 3, 4, 5)
    assert isinstance(pixels, numpy.ndarray) and pixels.shape == (2, 3, 4, 5)
    out = unwrap([{"a": 1}, {"a": 2}])
    assert isinstance(out, list) and len(out) == 2  # batch -> list


def test_list_of_2d_is_batch_unit_z():
    imgs = [_img((4, 5), i) for i in range(3)]
    masks = [numpy.ones((4, 5), int) for _ in range(3)]
    m, p, unwrap = to_bzyx(masks, imgs)
    assert len(m) == 3
    assert all(a.shape == (1, 4, 5) for a in m)
    assert isinstance(unwrap([1, 2, 3]), list)


def test_list_ragged_sizes_raise():
    imgs = [_img((4, 5), 0), _img((7, 3), 1)]
    masks = [numpy.ones((4, 5), int), numpy.ones((7, 3), int)]
    with pytest.raises(ValueError, match="same shape"):
        to_bzyx(masks, imgs)


def test_list_of_3d_volumes_batch():
    vols = [_img((2, 4, 5), 0), _img((2, 4, 5), 1)]
    masks = [numpy.ones((2, 4, 5), int), numpy.ones((2, 4, 5), int)]
    m, _, _ = to_bzyx(masks, vols)
    assert len(m) == 2 and all(a.shape == (2, 4, 5) for a in m)


@pytest.mark.parametrize(
    "masks, pixels, match",
    [
        (
            [numpy.ones((4, 5), int)],
            [_img((4, 5), 0), _img((4, 5), 1)],
            "batch size mismatch",
        ),
        ([numpy.ones((4, 5), int)], _img((4, 5), 0), "both be sequences"),
        (numpy.ones((2, 3, 4, 5), int), _img((3, 4, 5), 0), "both be 4D"),
        (numpy.ones((5,), int), numpy.ones((5,), float), "2D or 3D"),
    ],
)
@pytest.mark.parametrize("normalize", [to_bzyx, each_volume])
def test_normalizer_raises(masks, pixels, match, normalize):
    with pytest.raises(ValueError, match=match):
        normalize(masks, pixels)


@pytest.mark.parametrize("shape", [(4, 5), (1, 4, 5), (2, 4, 5), (2, 1, 4, 5)])
def test_each_volume_preserves_views_and_original_rank(shape):
    masks = numpy.ones(shape, numpy.int32)[..., ::-1]
    pixels = _img(shape, 0)[..., ::-1]
    masks.setflags(write=False)
    pixels.setflags(write=False)
    items, unwrap = each_volume(masks, pixels)
    sources = list(zip(masks, pixels)) if len(shape) == 4 else [(masks, pixels)]
    assert len(items) == len(sources)
    for (mask, image, rank), (original_mask, original_image) in zip(items, sources):
        assert rank == original_image.ndim
        assert (
            mask.shape
            == image.shape
            == ((1, *original_image.shape) if rank == 2 else original_image.shape)
        )
        assert numpy.shares_memory(mask, original_mask)
        assert numpy.shares_memory(image, original_image)
        assert not mask.flags.writeable and not image.flags.writeable
    results = list(range(len(items)))
    assert unwrap(results) == (results if len(shape) == 4 else results[0])


def test_each_volume_mixed_dtype_sequence_keeps_item_dtypes():
    masks = [numpy.ones((4, 5), numpy.int32) for _ in range(2)]
    pixels = [_img((4, 5), 0).astype(dtype) for dtype in (numpy.float32, numpy.float64)]
    items, unwrap = each_volume(masks, pixels)
    assert len(items) == 2
    for (_, image, rank), original in zip(items, pixels):
        assert rank == 2 and image.dtype == original.dtype
        assert numpy.shares_memory(image, original)
    assert unwrap([0, 1]) == [0, 1]
