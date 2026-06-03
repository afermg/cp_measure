"""Tests for the canonical (B,Z,Y,X) input normalisation helper."""

import numpy
import pytest

from cp_measure.primitives.shapes import to_bzyx


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
    assert len(masks) == len(pixels) == 2
    assert all(a.shape == (3, 4, 5) for a in masks)
    out = unwrap([{"a": 1}, {"a": 2}])
    assert isinstance(out, list) and len(out) == 2  # batch -> list


def test_list_of_2d_is_batch_unit_z():
    imgs = [_img((4, 5), i) for i in range(3)]
    masks = [numpy.ones((4, 5), int) for _ in range(3)]
    m, p, unwrap = to_bzyx(masks, imgs)
    assert len(m) == 3
    assert all(a.shape == (1, 4, 5) for a in m)
    assert isinstance(unwrap([1, 2, 3]), list)


def test_list_ragged_sizes_ok():
    imgs = [_img((4, 5), 0), _img((7, 3), 1)]
    masks = [numpy.ones((4, 5), int), numpy.ones((7, 3), int)]
    m, p, _ = to_bzyx(masks, imgs)
    assert m[0].shape == (1, 4, 5)
    assert m[1].shape == (1, 7, 3)


def test_list_of_3d_volumes_batch():
    vols = [_img((2, 4, 5), 0), _img((3, 4, 5), 1)]
    masks = [numpy.ones((2, 4, 5), int), numpy.ones((3, 4, 5), int)]
    m, _, _ = to_bzyx(masks, vols)
    assert m[0].shape == (2, 4, 5) and m[1].shape == (3, 4, 5)


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
def test_to_bzyx_raises(masks, pixels, match):
    with pytest.raises(ValueError, match=match):
        to_bzyx(masks, pixels)
