"""Singleton-Z eligibility is shared by all 2D-only measurement families.

3D-capable measurements keep their declared rank; a batch axis is never Z.
NumPy raw functions remain per-image APIs; Numba additionally accepts batches.
"""

import warnings

import numpy as np
import pytest

import cp_measure
from cp_measure.bulk import get_core_measurements, get_correlation_measurements
from cp_measure.featurizer import featurize, make_featurizer_config
from conftest import ALL_OFF

TWO_D_ONLY = ("radial_distribution", "radial_zernikes", "zernike", "feret")


def scene():
    masks = np.zeros((32, 40), dtype=np.int32)
    masks[3:13, 4:17] = 7
    masks[17:28, 21:35] = 91
    pixels = np.random.default_rng(19).random(masks.shape) * 0.8 + 0.1
    return masks, pixels


@pytest.fixture(params=[None, "numba"], ids=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    previous = cp_measure._ACCELERATOR
    cp_measure.set_accelerator(request.param)
    try:
        yield request.param
    finally:
        cp_measure.set_accelerator(previous)


def assert_same(got, expected):
    assert list(got) == list(expected)
    for key in expected:
        assert type(got[key]) is type(expected[key])
        actual_values, expected_values = np.asarray(got[key]), np.asarray(expected[key])
        assert actual_values.shape == expected_values.shape
        assert actual_values.dtype == expected_values.dtype
        np.testing.assert_array_equal(actual_values, expected_values, err_msg=key)


@pytest.mark.parametrize("feature", TWO_D_ONLY)
def test_singleton_z_is_eligible_without_mutating_inputs(backend, feature):
    masks, pixels = scene()
    # Negative strides and gapped labels exercise the public sanitation boundary.
    masks, pixels = masks[None, ::-1, ::-1], pixels[None, ::-1, ::-1]
    original_masks, original_pixels = masks.copy(), pixels.copy()
    fn = get_core_measurements()[feature]
    assert_same(fn(masks, pixels), fn(masks[0], pixels[0]))
    np.testing.assert_array_equal(masks, original_masks)
    np.testing.assert_array_equal(pixels, original_pixels)


@pytest.mark.parametrize("feature", TWO_D_ONLY)
def test_multislice_is_ineligible_even_when_all_slices_are_equal(backend, feature):
    masks, pixels = scene()
    fn = get_core_measurements()[feature]
    assert (
        fn(np.repeat(masks[None], 2, axis=0), np.repeat(pixels[None], 2, axis=0)) == {}
    )


@pytest.mark.parametrize("feature", TWO_D_ONLY)
@pytest.mark.parametrize(
    "container", [list, tuple, np.stack], ids=["list", "tuple", "stacked"]
)
@pytest.mark.parametrize("depth", [1, 3])
def test_numba_batch_matches_individual_volumes(feature, container, depth):
    pytest.importorskip("numba")
    previous = cp_measure._ACCELERATOR
    try:
        cp_measure.set_accelerator("numba")
        fn = get_core_measurements()[feature]
        masks, pixels = scene()
        masks = np.repeat(masks[None], depth, axis=0)
        pixels = np.repeat(pixels[None], depth, axis=0)
        # Distinct image contents and object counts must not leak across B.
        other = np.where(masks == 7, 300, 0).astype(np.int32)
        ms, ps = [masks, other], [pixels, pixels * 2]
        got = fn(container(ms), container(ps))
        assert isinstance(got, list) and len(got) == 2
        for result, m, p in zip(got, ms, ps):
            assert_same(result, fn(m, p))
    finally:
        cp_measure.set_accelerator(previous)


@pytest.mark.parametrize("feature", ["radial_distribution", "radial_zernikes"])
@pytest.mark.parametrize("reverse", [False, True])
def test_radial_does_not_broadcast_singleton_spatial_axes(backend, feature, reverse):
    masks, pixels = scene()
    if reverse:
        pixels = pixels[None]
    else:
        masks = masks[None]
    with pytest.raises(ValueError, match="matching shapes"):
        get_core_measurements()[feature](masks, pixels)


@pytest.mark.parametrize("feature", TWO_D_ONLY)
def test_featurizer_singleton_z_matches_2d_without_skip_warning(backend, feature):
    masks, pixels = scene()
    masks, pixels = masks[None, None], pixels[None, None]
    config = make_featurizer_config(["DNA"], **{**ALL_OFF, feature: True})
    expected = featurize(pixels, masks, config, image_ids=["image"])
    original_masks, original_pixels = masks.copy(), pixels.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = featurize(
            pixels[:, :, None],
            masks[:, :, None],
            config,
            is_3d=True,
            image_ids=["image"],
        )
    assert not any("skipping 2D-only" in str(w.message) for w in caught)
    assert got[1:] == expected[1:]
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(masks, original_masks)
    np.testing.assert_array_equal(pixels, original_pixels)


def test_featurizer_multislice_skips_only_ineligible_families(backend):
    masks, pixels = scene()
    masks = np.repeat(masks[None, None, None], 3, axis=2)
    pixels = np.repeat(pixels[None, None, None], 3, axis=2)
    config = make_featurizer_config(
        ["DNA"], **{**ALL_OFF, "intensity": True, **dict.fromkeys(TWO_D_ONLY, True)}
    )
    with pytest.warns(UserWarning, match="skipping 2D-only"):
        got = featurize(pixels, masks, config, is_3d=True)
    config = make_featurizer_config(["DNA"], **{**ALL_OFF, "intensity": True})
    expected = featurize(pixels, masks, config, is_3d=True)
    assert got[1:] == expected[1:]
    np.testing.assert_array_equal(got[0], expected[0])


@pytest.mark.parametrize("feature", ["intensity", "texture", "granularity"])
def test_3d_capable_families_preserve_rank_through_featurizer(backend, feature):
    from cp_measure.core import (
        measuregranularity,
        measureobjectintensity,
        measuretexture,
    )

    masks, pixels = scene()
    masks = np.where(masks == 7, 1, np.where(masks == 91, 2, 0)).astype(np.int32)[None]
    pixels = pixels[None]
    ref_fn = {
        "intensity": measureobjectintensity.get_intensity,
        "texture": measuretexture.get_texture,
        "granularity": measuregranularity.get_granularity,
    }[feature]
    # A backend may differ in floating-point reduction order; preserve the
    # project's existing comparison tolerance, not blanket bitwise parity.
    expected = ref_fn(masks, pixels)
    got = get_core_measurements()[feature](masks, pixels)
    assert list(got) == list(expected)
    for key in expected:
        np.testing.assert_allclose(
            got[key], expected[key], rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=key
        )
    config = make_featurizer_config(["DNA"], **{**ALL_OFF, feature: True})
    output = featurize(pixels[None, None], masks[None, None], config, is_3d=True)
    np.testing.assert_allclose(
        output[0],
        np.column_stack(list(got.values())),
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "container", [list, tuple, np.stack], ids=["list", "tuple", "stacked"]
)
def test_granularity_singleton_z_batch_preserves_numpy_3d_behavior(container):
    pytest.importorskip("numba")
    from cp_measure.core.measuregranularity import get_granularity as ref
    from cp_measure.core.numba.measuregranularity import get_granularity as got

    masks, pixels = scene()
    masks = np.where(masks == 7, 1, np.where(masks == 91, 2, 0)).astype(np.int32)[None]
    pixels = pixels[None]
    for result in got(container([masks, masks]), container([pixels, pixels])):
        assert_same(result, ref(masks, pixels))


def test_singleton_z_does_not_reinterpret_3d_sizeshape(backend):
    masks, pixels = scene()
    with pytest.raises(ValueError, match="at least 2x2x2"):
        get_core_measurements()["sizeshape"](masks[None], pixels[None])


@pytest.mark.parametrize("feature", ["pearson", "manders_fold", "rwc", "costes"])
def test_correlation_singleton_z_preserves_2d_values(backend, feature):
    masks, pixels = scene()
    second = np.random.default_rng(5).random(masks.shape) * 0.8 + 0.1
    fn = get_correlation_measurements()[feature]
    kwargs = {"scale_max": 65535} if feature == "costes" else {}
    assert_same(
        fn(pixels[None], second[None], masks[None], **kwargs),
        fn(pixels, second, masks, **kwargs),
    )
