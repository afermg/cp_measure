"""Shared contracts for batch-capable Numba APIs, not per-image-only APIs.

Numerical NumPy parity remains in family tests. Here batch items must equal
independent calls, preserving rank, order, label isolation and caller arrays.
Shape-only APIs ignore pixels; empty-batch and shape-validation policies are
not uniform, so the rejection checks below name the applicable capabilities.
"""

from functools import partial
from importlib import import_module
from inspect import Parameter, signature

import numpy as np
import pytest

from cp_measure._detect import HAS_NUMBA
from cp_measure._sanitize import sanitize
from cp_measure.bulk import _numba_registries

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
CORE = [
    "radial_distribution",
    "radial_zernikes",
    "zernike",
    "feret",
    "texture",
    "granularity",
]
CORRELATION = ["pearson", "manders_fold", "rwc", "costes", "overlap"]
CASES = [(name, {}) for name in CORE + CORRELATION] + [
    ("intensity", dict(legacy=legacy, edge_measurements=edge))
    for legacy in (False, True)
    for edge in (False, True)
]


@pytest.fixture(params=CASES, ids=[name + str(options) for name, options in CASES])
def measurement(request):
    name, options = request.param
    registries = _numba_registries()
    if name == "overlap":
        from cp_measure.core.numba.measurecolocalization import get_correlation_overlap

        function = get_correlation_overlap
    else:
        function = registries["correlation" if name in CORRELATION else "core"][name]
    # Small full-resolution morphology avoids unrelated subsampled 3D behavior.
    if name == "granularity":
        options = dict(
            subsample_size=1,
            image_sample_size=1,
            element_size=2,
            granular_spectrum_length=3,
        )
    elif name == "texture":
        options = dict(scale=1, gray_levels=8)
    elif name == "costes":
        options = dict(scale_max=255)
    function = partial(sanitize(function), **options)

    def call(mask, first, second):
        return (
            function(first, second, mask)
            if name in CORRELATION
            else function(mask, first)
        )

    reference = None
    if name == "intensity":
        from cp_measure.core.measureobjectintensity import get_intensity

        reference = partial(sanitize(get_intensity), **options)
    return name, call, reference


@pytest.mark.parametrize(
    "module, helper",
    [
        ("measureobjectintensity", "_intensity_volume"),
        ("measuregranularity", "_granularity_one"),
    ],
)
def test_original_rank_metadata_is_required(module, helper):
    function = getattr(import_module("cp_measure.core.numba." + module), helper)
    parameter = signature(function).parameters["spatial_ndim"]
    assert parameter.kind is Parameter.KEYWORD_ONLY
    assert parameter.default is Parameter.empty


def scenes(depth):
    rng = np.random.default_rng(57127)
    masks = [np.zeros((24, 24), np.int32) for _ in range(2)]
    masks[0][2:10, 3:11] = 1
    masks[0][13:22, 12:23] = 2
    masks[1][4:17, 5:19] = 1
    shape = (24, 24) if depth == 0 else (depth, 24, 24)
    if depth:
        masks = [np.repeat(m[None], depth, axis=0) for m in masks]
    first = [rng.random(shape) for _ in masks]
    second = [0.65 * p + 0.35 * rng.random(shape) for p in first]
    return masks, first, second


def assert_same(actual, expected):
    assert isinstance(actual, dict) and list(actual) == list(expected)
    for key in expected:
        a, e = np.asarray(actual[key]), np.asarray(expected[key])
        assert a.dtype == e.dtype and a.shape == e.shape
        np.testing.assert_array_equal(a, e, err_msg=key)


@pytest.mark.parametrize("depth", [0, 1, 3], ids=["2d", "singleton-z", "3d"])
@pytest.mark.parametrize(
    "container", [list, tuple, np.stack], ids=["list", "tuple", "stacked"]
)
@pytest.mark.parametrize("labels", ["readonly", "gapped"])
def test_batch_matches_independent_items(measurement, depth, container, labels):
    name, call, reference = measurement
    inputs = scenes(depth)
    if labels == "gapped":
        inputs[0][0][inputs[0][0] == 2] = 7
        inputs[0][1][inputs[0][1] == 1] = 42
    if name in CORRELATION and name != "costes":
        # Preserve tied-rank coverage without making intensity maxima ambiguous.
        inputs[1][1] = np.floor(inputs[1][1] * 8) / 8
        inputs[2][1] = np.floor(inputs[2][1] * 8) / 8
    if container is np.stack and depth == 0:
        inputs = tuple([a[None] for a in group] for group in inputs)
    inputs = tuple(container(group) for group in inputs)
    originals = [[a.copy() for a in group] for group in inputs]
    # Older skimage relabel_sequential rejects readonly sparse labels. Exercise
    # readonly dense inputs and non-mutating sparse sanitation separately.
    for i, group in enumerate(inputs):
        if i == 0 and labels == "gapped":
            continue
        for array in [group] if isinstance(group, np.ndarray) else group:
            array.setflags(write=False)
    expected = [call(*item) for item in zip(*inputs)]
    actual = call(*inputs)
    assert isinstance(actual, list) and len(actual) == len(expected)
    for got, ref, item in zip(actual, expected, zip(*inputs)):
        assert_same(got, ref)
        if reference is not None:
            numpy_result = reference(item[0], item[1])
            assert list(got) == list(numpy_result)
            for key in got:
                assert got[key].dtype == numpy_result[key].dtype
                np.testing.assert_allclose(
                    got[key], numpy_result[key], rtol=1e-10, atol=1e-12
                )
    for group, before in zip(inputs, originals):
        for a, b in zip(group, before):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("case", ["count", "mixed", "ragged"])
def test_invalid_paired_batch_structure(measurement, case):
    name, call, _ = measurement
    if name in ("zernike", "feret"):
        pytest.skip("shape-only API intentionally ignores pixels")
    masks, first, second = scenes(0)
    if case == "count":
        first = first[:1]
    elif case == "mixed":
        first = first[0]
    else:
        masks[1], first[1], second[1] = (
            a[:12] for a in (masks[1], first[1], second[1])
        )
    with pytest.raises(ValueError):
        call(masks, first, second)


@pytest.mark.parametrize("shape", [(24, 25), (1, 24, 24), (3, 24, 24)])
@pytest.mark.parametrize("container", ["single", "list", "tuple", "stacked"])
def test_measured_image_shape_mismatch_rejected(
    measurement, shape, container, monkeypatch
):
    name, call, _ = measurement
    if name in ("zernike", "feret", "granularity"):
        pytest.skip("shape-only pixels ignored; granularity has no shared shape guard")
    if name == "intensity":
        from cp_measure.core.numba import measureobjectintensity

        def forbidden(*args, **kwargs):
            pytest.fail("invalid inputs reached unchecked intensity kernel")

        monkeypatch.setattr(measureobjectintensity, "_intensity_volume", forbidden)
    masks, _, _ = scenes(0)
    inputs = (masks[0], np.ones(shape), np.ones(shape))
    if container in ("list", "tuple"):
        factory = list if container == "list" else tuple
        inputs = tuple(factory([a]) for a in inputs)
    elif container == "stacked":
        inputs = tuple(a.reshape((1, -1, *a.shape[-2:])) for a in inputs)
        if inputs[0].shape == inputs[1].shape:
            inputs = (inputs[0], *(np.repeat(a, 2, axis=1) for a in inputs[1:]))
    with pytest.raises(ValueError):
        call(*inputs)


def test_empty_batch_policy(measurement):
    name, call, _ = measurement
    if name in ("feret", "texture"):
        assert call([], [], []) == []
    else:
        with pytest.raises(ValueError):
            call([], [], [])
