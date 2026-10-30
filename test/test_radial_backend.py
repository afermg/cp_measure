"""Backend correctness for numba radial_distribution.

Parity with the numpy reference on multi-object fields (border objects, centre
plateaus, disconnected labels; zero and integer intensities; scaled and unscaled
bins), the Issue #22 property (object k in a field equals the reference on object
k in isolation), and the 2D-only / batch / invalid-input paths.
"""

import numpy as np
import pytest

from cp_measure._detect import HAS_NUMBA
from cp_measure.core.measureobjectintensitydistribution import (
    get_radial_distribution as ref,
)

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


def _numba():
    from cp_measure.core.numba.measureobjectintensitydistribution import (
        get_radial_distribution,
    )

    return get_radial_distribution


def _issue22_masks():
    """The Issue #22 example: two objects with edge asymmetries and even-sized
    squares, so each centre is a tied 2x2 plateau."""
    masks = np.zeros((240, 240), np.int32)
    masks[50:100, 50:100] = 1
    masks[80:120, 90:120] = 1
    masks[150:200, 150:200] = 2
    masks[175:180, 180:210] = 2
    return masks


def _border_masks():
    """Objects cut by the image border and corners, as in most fields of view."""
    masks = np.zeros((60, 60), np.int32)
    masks[0:21, 0:31] = 1
    masks[10:41, 50:60] = 2
    masks[40:60, 0:25] = 3
    return masks


def _disconnected_masks():
    """One label in two pieces: pixels unreachable from its centre are skipped."""
    masks = np.zeros((60, 60), np.int32)
    masks[5:30, 5:30] = 1
    masks[40:50, 40:50] = 1
    masks[35:55, 5:20] = 2
    return masks


def _pixels(kind, shape):
    rng = np.random.default_rng(42)
    if kind == "zeros":
        return np.zeros(shape)
    if kind == "uint16":
        return (rng.random(shape) * 4095).astype(np.uint16)
    return rng.random(shape)


def _assert_close(a, b, key=""):
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=key)


def _assert_close_dict(got, exp):
    assert list(got) == list(exp)
    for key in exp:
        assert got[key].shape == exp[key].shape
        assert got[key].dtype == exp[key].dtype
        _assert_close(got[key], exp[key], key)


@requires_numba
@pytest.mark.parametrize(
    "masks_fn", [_issue22_masks, _border_masks, _disconnected_masks]
)
@pytest.mark.parametrize("pixels_kind", ["random", "zeros", "uint16"])
@pytest.mark.parametrize(
    "scaled,maximum_radius", [(True, 100), (False, 10), (False, 20)]
)
def test_numba_matches_numpy(masks_fn, pixels_kind, scaled, maximum_radius):
    masks = masks_fn()
    pixels = _pixels(pixels_kind, masks.shape)
    kwargs = dict(scaled=scaled, maximum_radius=maximum_radius)
    with np.errstate(all="ignore"):
        _assert_close_dict(
            _numba()(masks, pixels, **kwargs), ref(masks, pixels, **kwargs)
        )


@requires_numba
@pytest.mark.parametrize("scaled", [True, False])
def test_numba_multi_equals_numpy_isolated(scaled):
    """Issue #22: an object's result does not depend on other labels."""
    masks = _issue22_masks()
    pixels = _pixels("random", masks.shape)
    got = _numba()(masks, pixels, scaled=scaled)
    for k, label in enumerate((1, 2)):
        iso = (masks == label).astype(np.int32)  # object alone, relabelled to 1
        exp = ref(iso, pixels, scaled=scaled)
        assert set(got) == set(exp)
        for key in exp:
            _assert_close(got[key][k], exp[key][0], key)


@requires_numba
@pytest.mark.parametrize("depth", [1, 3])
def test_3d_volume_returns_empty(depth):
    vol = np.zeros((depth, 32, 32), np.int32)
    vol[:, 5:15, 5:15] = 1
    pix = np.random.default_rng(0).random((depth, 32, 32))
    assert _numba()(vol, pix) == ref(vol, pix) == {}
    assert _numba()([vol], [pix]) == [{}]


@requires_numba
def test_empty_image_empty_arrays():
    masks = np.zeros((40, 40), np.int32)
    pix = np.random.default_rng(0).random((40, 40))
    got = _numba()(masks, pix)
    assert all(v.shape == (0,) for v in got.values())
    assert set(got) == set(ref(masks, pix))


@requires_numba
def test_batch_list_matches_per_image():
    m0 = _issue22_masks()
    m1 = (m0 == 1).astype(np.int32)
    p0 = _pixels("random", m0.shape)
    got = _numba()([m0, m1], [p0, p0])
    assert isinstance(got, list) and len(got) == 2
    for per_image, m in zip(got, (m0, m1)):
        _assert_close_dict(per_image, _numba()(m, p0))


@requires_numba
@pytest.mark.parametrize(
    "pixels_shape,kwargs",
    [
        ((50, 50), {}),
        ((3, 60, 60), {}),
        ((60, 60), dict(scaled=False, maximum_radius=-1)),
        ((60, 60), dict(bin_count=-1)),
    ],
    ids=["smaller-image", "channel-stack", "negative-radius", "negative-bins"],
)
def test_invalid_inputs_raise(pixels_shape, kwargs):
    """The kernel has no bounds checks: these would read or write out of bounds."""
    masks = np.zeros((60, 60), np.int32)
    masks[5:50, 5:50] = 1
    pixels = np.random.default_rng(0).random(pixels_shape)
    with pytest.raises(ValueError):
        _numba()(masks, pixels, **kwargs)


@requires_numba
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_singleton_intensity_axis_is_not_silently_promoted(as_list, empty):
    masks = _border_masks()
    if empty:
        masks[:] = 0
    pixels = _pixels("random", masks.shape)[None]
    with pytest.raises((IndexError, ValueError)):
        ref(masks, pixels)
    if as_list:
        masks, pixels = [masks], [pixels]
    with pytest.raises(ValueError, match="matching shapes"):
        _numba()(masks, pixels)


@requires_numba
@pytest.mark.parametrize("bin_count", [-1, -2, 4.5, -0.5])
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_bin_count_matches_numpy(bin_count, empty):
    masks = _border_masks()
    if empty:
        masks[:] = 0
    pixels = _pixels("random", masks.shape)
    error = TypeError if isinstance(bin_count, float) else ValueError
    for fn in (ref, _numba()):
        with pytest.raises(error):
            fn(masks, pixels, bin_count=bin_count)


@requires_numba
@pytest.mark.parametrize("maximum_radius", [0, np.nan])
def test_nonfinite_bin_indexes_raise_like_numpy(maximum_radius):
    masks = _border_masks()
    pixels = _pixels("random", masks.shape)
    with np.errstate(all="ignore"):
        for fn in (ref, _numba()):
            with pytest.raises(ValueError):
                fn(masks, pixels, scaled=False, maximum_radius=maximum_radius)


@requires_numba
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided", "reversed"])
@pytest.mark.parametrize("scaled", [True, False])
def test_layout_label_remapping_and_input_immutability(layout, scaled):
    from cp_measure import bulk, set_accelerator

    masks = _border_masks()
    masks = np.choose(masks, [0, 91, 7, 1003]).astype(np.int32)
    pixels = _pixels("random", masks.shape)
    if layout == "transpose":
        masks, pixels = masks.T, pixels.T
    elif layout == "strided":
        masks, pixels = masks[::2, ::2], pixels[::2, ::2]
    elif layout == "reversed":
        masks, pixels = masks[::-1, ::-1], pixels[::-1, ::-1]
    before = masks.copy(), pixels.copy()
    kwargs = dict(scaled=scaled, bin_count=5, maximum_radius=7)
    try:
        set_accelerator(None)
        expected = bulk.get_core_measurements()["radial_distribution"](
            labels=masks, pixels=pixels, **kwargs
        )
        set_accelerator("numba")
        got = bulk.get_core_measurements()["radial_distribution"](
            labels=masks, pixels=pixels, **kwargs
        )
    finally:
        set_accelerator(None)
    assert list(got) == list(expected)
    for key in got:
        assert got[key].dtype == expected[key].dtype == np.float64
        assert got[key].shape == (3,)
    _assert_close_dict(got, expected)
    for i, label in enumerate((7, 91, 1003)):
        isolated = ref((masks == label).astype(np.int32), pixels, **kwargs)
        for key in got:
            _assert_close(got[key][i], isolated[key][0], key)
    np.testing.assert_array_equal(masks, before[0])
    np.testing.assert_array_equal(pixels, before[1])


@requires_numba
@pytest.mark.parametrize("depth", [1, 3])
def test_stacked_batch_volume_contract(depth):
    masks = np.repeat(_border_masks()[None, None], 2, axis=0)
    masks = np.repeat(masks, depth, axis=1)
    pixels = _pixels("random", masks.shape)
    got = _numba()(labels=masks, pixels=pixels)
    if depth > 1:
        assert got == [{}, {}]
    else:
        for i in range(2):
            _assert_close_dict(got[i], ref(masks[i, 0], pixels[i, 0]))


@requires_numba
@pytest.mark.parametrize("pixels_kind", ["signed", "nan", "inf", "float32"])
@pytest.mark.parametrize("scaled", [True, False])
def test_nonfinite_signed_and_float32_pixels(pixels_kind, scaled):
    masks = _disconnected_masks()
    pixels = _pixels("random", masks.shape)
    if pixels_kind == "signed":
        pixels -= 0.5
    elif pixels_kind == "float32":
        pixels = pixels.astype(np.float32)
    else:
        pixels[::7, ::7] = np.nan if pixels_kind == "nan" else np.inf
    with np.errstate(all="ignore"):
        kwargs = dict(scaled=scaled, bin_count=9, maximum_radius=7)
        _assert_close_dict(
            _numba()(masks, pixels, **kwargs), ref(masks, pixels, **kwargs)
        )


@requires_numba
def test_featurizer_radial_label_ids_columns_and_immutability():
    from conftest import ALL_OFF

    from cp_measure import set_accelerator
    from cp_measure.featurizer import featurize, make_featurizer_config

    masks = _border_masks()
    masks = np.choose(masks, [0, 91, 7, 1003]).astype(np.int32)
    pixels = _pixels("random", masks.shape)
    images = np.stack([pixels, pixels * 2])[:, None]
    labels = np.stack([masks, masks])[:, None]
    before = images.copy(), labels.copy()
    config = make_featurizer_config(["DNA"], **{**ALL_OFF, "radial_distribution": True})
    try:
        set_accelerator(None)
        expected = featurize(images, labels, config, image_ids=["a", "b"])
        set_accelerator("numba")
        got = featurize(images, labels, config, image_ids=["a", "b"])
    finally:
        set_accelerator(None)
    assert got[1] == expected[1]
    assert len(got[1]) == 12
    assert (
        got[2]
        == expected[2]
        == [(image, "object", label) for image in ("a", "b") for label in (7, 91, 1003)]
    )
    _assert_close(got[0], expected[0])
    np.testing.assert_array_equal(images, before[0])
    np.testing.assert_array_equal(labels, before[1])


@requires_numba
@pytest.mark.parametrize("bin_count", [0, np.int32(4)])
@pytest.mark.parametrize("scaled,maximum_radius", [(True, None), (False, np.inf)])
def test_valid_bin_scalars_and_unused_radius(bin_count, scaled, maximum_radius):
    masks = _border_masks()
    pixels = _pixels("random", masks.shape)
    kwargs = dict(bin_count=bin_count, scaled=scaled, maximum_radius=maximum_radius)
    with np.errstate(all="ignore"):
        _assert_close_dict(
            _numba()(masks, pixels, **kwargs), ref(masks, pixels, **kwargs)
        )
