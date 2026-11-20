"""Default boundary inclusion and explicit pre-PR Zernike compatibility."""

from importlib import import_module

import numpy as np
import pytest
from conftest import ALL_OFF
from test_radial_zernike import _reference as radial_reference
from test_zernike import _reference as shape_reference

import cp_measure
from cp_measure.featurizer import featurize, make_featurizer_config


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


@pytest.fixture(params=["zernike", "radial_zernikes"])
def feature(request):
    return request.param


@pytest.fixture
def measurement(backend, feature):
    family = (
        "measureobjectsizeshape"
        if feature == "zernike"
        else "measureobjectintensitydistribution"
    )
    prefix = "cp_measure.core." + ("numba." if backend == "numba" else "")
    return getattr(import_module(prefix + family), "get_" + feature)


@pytest.fixture
def scene():
    masks = np.zeros((24, 32), dtype=np.int32)
    masks[3:7, 5:9] = 1
    masks[12:19, 20:27] = 2
    pixels = np.random.default_rng(137).random(masks.shape)
    return masks, pixels


def assert_features_equal(actual, expected, *, exact=False):
    assert list(actual) == list(expected)
    for name, expected_values in expected.items():
        actual_values = actual[name]
        assert actual_values.shape == expected_values.shape
        assert actual_values.dtype == expected_values.dtype
        if exact:
            np.testing.assert_array_equal(actual_values, expected_values, err_msg=name)
        elif "Phase" not in name:
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=1e-9,
                atol=1e-10,
                equal_nan=True,
                err_msg=name,
            )
        else:
            np.testing.assert_array_equal(
                np.isnan(actual_values), np.isnan(expected_values)
            )
            magnitude = expected[name.replace("Phase", "Magnitude")]
            scale = expected["RadialDistribution_ZernikeMagnitude_0_0"]
            defined = np.isfinite(magnitude) & (magnitude > 1e-9 * scale)
            difference = np.angle(
                np.exp(1j * (actual_values[defined] - expected_values[defined]))
            )
            np.testing.assert_allclose(difference, 0, atol=1e-7, err_msg=name)


@pytest.mark.parametrize("degree", [0, 5, 9])
def test_legacy_matches_unmodified_centrosome(measurement, feature, scene, degree):
    masks, pixels = scene
    expected = (
        shape_reference(masks, degree, legacy_boundary=True)
        if feature == "zernike"
        else radial_reference(masks, pixels, degree, legacy_boundary=True)
    )
    actual = measurement(masks, pixels, degree, legacy_boundary=True)
    assert_features_equal(actual, expected)


def test_default_is_corrected_and_legacy_changes_boundary_membership(
    measurement,
    feature,
    scene,
):
    masks, pixels = scene
    corrected = measurement(masks, pixels, legacy_boundary=False)
    assert_features_equal(measurement(masks, pixels), corrected, exact=True)
    legacy = measurement(masks, pixels, legacy_boundary=True)
    name = (
        "Zernike_0_0"
        if feature == "zernike"
        else "RadialDistribution_ZernikeMagnitude_0_0"
    )
    assert np.all(corrected[name] > legacy[name])


@pytest.mark.parametrize("legacy_boundary", [False, True])
@pytest.mark.parametrize("container", [list, tuple, np.stack])
def test_batch_shared_mask_and_mode_isolation(
    measurement,
    scene,
    legacy_boundary,
    container,
):
    masks, pixels = scene
    masks = masks[None]
    pixels = pixels[None]
    other_pixels = pixels * 0.5
    for array in (masks, pixels, other_pixels):
        array.flags.writeable = False
    mask_batch = container([masks, masks])
    pixel_batch = container([pixels, other_pixels])
    actual = measurement(mask_batch, pixel_batch, legacy_boundary=legacy_boundary)
    measurement(mask_batch, pixel_batch, legacy_boundary=not legacy_boundary)
    repeated = measurement(mask_batch, pixel_batch, legacy_boundary=legacy_boundary)
    assert isinstance(actual, list) and len(actual) == 2
    for index, image in enumerate((pixels, other_pixels)):
        expected = measurement(masks, image, legacy_boundary=legacy_boundary)
        assert_features_equal(actual[index], expected, exact=True)
        assert_features_equal(repeated[index], expected, exact=True)
    for name in actual[0]:
        assert not np.shares_memory(actual[0][name], actual[1][name])


@pytest.mark.parametrize("shape", [(0, 0), (8, 8), (1, 8, 8), (2, 8, 8)])
@pytest.mark.parametrize("legacy_boundary", [False, True])
def test_empty_and_volume_contracts(measurement, shape, legacy_boundary):
    masks = np.zeros(shape, dtype=np.int32)
    pixels = np.zeros(shape)
    result = measurement(masks, pixels, legacy_boundary=legacy_boundary)
    if len(shape) == 3 and shape[0] > 1:
        assert result == {}
    else:
        assert result and all(values.shape == (0,) for values in result.values())


@pytest.mark.parametrize("legacy_boundary", [False, True])
def test_featurizer_forwards_boundary_policy(
    backend,
    feature,
    measurement,
    scene,
    legacy_boundary,
):
    masks, pixels = scene
    expected = measurement(masks, pixels, legacy_boundary=legacy_boundary)
    previous = cp_measure._ACCELERATOR
    try:
        cp_measure.set_accelerator(None if backend == "numpy" else "numba")
        config = make_featurizer_config(
            ["DNA"],
            **{
                **ALL_OFF,
                feature: True,
                feature + "_params": {"legacy_boundary": legacy_boundary},
            },
        )
        values, columns, rows = featurize(
            pixels[None, None],
            masks[None, None],
            config,
        )
    finally:
        cp_measure.set_accelerator(previous)
    assert len(rows) == 2
    assert len(columns) == len(expected)
    np.testing.assert_allclose(
        values, np.column_stack(list(expected.values())), equal_nan=True
    )


def test_legacy_numba_restores_centrosome_coordinates(monkeypatch, scene):
    pytest.importorskip("numba")
    kernels = import_module("cp_measure.core.numba._zernike")
    shape = import_module("cp_measure.core.numba.measureobjectsizeshape").get_zernike
    radial = import_module(
        "cp_measure.core.numba.measureobjectintensitydistribution"
    ).get_radial_zernikes

    def fail_new_coordinates(*args, **kwargs):
        raise AssertionError("legacy mode must use the historical circle calculation")

    monkeypatch.setattr(kernels, "_shape_sums", fail_new_coordinates)
    monkeypatch.setattr(kernels, "_weighted_coords", fail_new_coordinates)
    shape(*scene, legacy_boundary=True)
    radial(*scene, legacy_boundary=True)


@pytest.mark.parametrize("legacy_boundary", [False, True])
def test_kernel_cutoff_edges(legacy_boundary):
    pytest.importorskip("numba")
    kernels = import_module("cp_measure.core.numba._zernike")
    threshold = 1.0 if legacy_boundary else 1.0 + 1e-9
    columns = np.array([1.0, np.nextafter(1.0, np.inf), 1.0 + 1e-10, 1.0 + 1e-8])
    rows = np.zeros(len(columns))
    segments = np.arange(len(columns))
    weights = np.arange(1.0, len(columns) + 1.0)
    lookup = np.ones((1, 1))
    terms = np.ones(1, dtype=np.int64)
    azimuths = np.zeros(1, dtype=np.int64)
    expected = np.where(columns**2 <= threshold, weights, 0)[:, None]
    real, imaginary = kernels.zernike_moments(
        weights,
        columns,
        rows,
        segments,
        lookup,
        terms,
        azimuths,
        len(columns),
        unit_disk_edge=threshold,
    )
    np.testing.assert_array_equal(real, expected)
    np.testing.assert_array_equal(imaginary, np.zeros_like(expected))
    real, imaginary = kernels.zernike_moments_channels(
        np.stack([weights, weights * 2]),
        columns,
        rows,
        segments,
        lookup,
        terms,
        azimuths,
        len(columns),
        unit_disk_edge=threshold,
    )
    np.testing.assert_array_equal(real, np.stack([expected, expected * 2]))
    np.testing.assert_array_equal(imaginary, np.zeros_like(real))


@pytest.mark.parametrize("legacy_boundary", [False, True])
def test_threaded_modes_match_serial(feature, scene, legacy_boundary):
    pytest.importorskip("numba")
    family = (
        "measureobjectsizeshape"
        if feature == "zernike"
        else "measureobjectintensitydistribution"
    )
    measurement = getattr(
        import_module("cp_measure.core.numba." + family), "get_" + feature
    )
    masks, pixels = scene
    mask_batch = [masks, np.roll(masks, 1, axis=0), masks]
    pixel_batch = [pixels, np.roll(pixels, 1, axis=0), pixels * 0.5]
    for mask, image in zip(mask_batch, pixel_batch):
        measurement(mask, image, legacy_boundary=legacy_boundary)
    actual = measurement(
        mask_batch,
        pixel_batch,
        legacy_boundary=legacy_boundary,
        max_workers=2,
    )
    for index, (mask, image) in enumerate(zip(mask_batch, pixel_batch)):
        expected = measurement(mask, image, legacy_boundary=legacy_boundary)
        assert_features_equal(actual[index], expected, exact=True)
