"""Backend correctness harness: a numba backend must match its numpy counterpart.

Compares ``intensity`` key-by-key on continuous random pixels (no exact ties, so
``Location_MaxIntensity_*`` is unambiguous), for 2D and 3D input with edge
measurements on and off; per-feature value comparisons for the other backends
live in their own ``test_<feature>_backend`` modules. Also checks the dispatch
wiring for every ported feature: under ``set_accelerator("numba")`` the core
registry composes the numba implementations with the numpy ones for everything
else, and an absent numba backend raises rather than silently falling back.
"""

import numpy as np
import pytest
from conftest import (
    DEPTH_3D,
    SIZE_2D,
    SIZE_3D,
    _stamp_objects_2d,
    _stamp_objects_3d,
    get_rng,
)

import cp_measure
import cp_measure._detect
import cp_measure.bulk
from cp_measure._detect import HAS_NUMBA
from cp_measure.core.measureobjectintensity import get_intensity as intensity_numpy

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


def _mask_pixels_2d():
    mask = np.zeros((SIZE_2D, SIZE_2D), dtype=np.int32)
    _stamp_objects_2d(mask, n_objects=3)
    return mask, get_rng().random((SIZE_2D, SIZE_2D))


def _mask_pixels_3d():
    mask = np.zeros((DEPTH_3D, SIZE_3D, SIZE_3D), dtype=np.int32)
    _stamp_objects_3d(mask, n_objects=2)
    return mask, get_rng().random((DEPTH_3D, SIZE_3D, SIZE_3D))


def _assert_dicts_match(ref, got):
    assert set(got) == set(ref), set(got).symmetric_difference(ref)
    for key in ref:
        np.testing.assert_allclose(
            got[key], ref[key], rtol=1e-6, atol=1e-8, err_msg=f"feature {key!r}"
        )


@requires_numba
@pytest.mark.parametrize("legacy", [False, True], ids=["new", "legacy"])
@pytest.mark.parametrize("edge", [True, False], ids=["edge", "noedge"])
@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_numba_intensity_matches_numpy(dim, edge, legacy):
    from cp_measure.core.numba.measureobjectintensity import (
        get_intensity as intensity_numba,
    )

    mask, pixels = _mask_pixels_2d() if dim == "2d" else _mask_pixels_3d()
    ref = intensity_numpy(mask, pixels, edge_measurements=edge, legacy=legacy)
    got = intensity_numba(mask, pixels, edge_measurements=edge, legacy=legacy)
    _assert_dicts_match(ref, got)


def test_legacy_flag_anchor():
    """Pin both backends to known numerical conventions on a tiny anchor case."""
    mask = np.array([[1, 1, 1, 1]], dtype=np.int32)
    pixels = np.array([[0.0, 1.0, 2.0, 3.0]])
    new = intensity_numpy(mask, pixels, edge_measurements=False, legacy=False)
    legacy = intensity_numpy(mask, pixels, edge_measurements=False, legacy=True)
    # np.percentile defaults
    np.testing.assert_allclose(new["Intensity_LowerQuartileIntensity"], [0.75])
    np.testing.assert_allclose(new["Intensity_MedianIntensity"], [1.5])
    np.testing.assert_allclose(new["Intensity_UpperQuartileIntensity"], [2.25])
    # legacy: pos = n * q on a sorted segment
    np.testing.assert_allclose(legacy["Intensity_LowerQuartileIntensity"], [1.0])
    np.testing.assert_allclose(legacy["Intensity_MedianIntensity"], [2.0])
    np.testing.assert_allclose(legacy["Intensity_UpperQuartileIntensity"], [3.0])


@requires_numba
@pytest.mark.parametrize("sub", [1.0, 0.25], ids=["fullres", "subsampled"])
def test_numba_granularity_matches_numpy(sub):
    from cp_measure.core.measuregranularity import get_granularity as gran_numpy
    from cp_measure.core.numba.measuregranularity import (
        get_granularity as gran_numba,
    )

    mask, pixels = _mask_pixels_2d()
    ref = gran_numpy(mask, pixels, subsample_size=sub, image_sample_size=sub)
    got = gran_numba(mask, pixels, subsample_size=sub, image_sample_size=sub)
    # bit-exact morphology; per-object mean is a float reduction -> small tol, nan-aware
    assert set(got) == set(ref)
    for key in ref:
        np.testing.assert_allclose(
            got[key], ref[key], rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=key
        )


@requires_numba
def test_numba_intensity_single_returns_dict():
    from cp_measure.core.numba.measureobjectintensity import (
        get_intensity as intensity_numba,
    )

    mask, pixels = _mask_pixels_2d()
    assert isinstance(intensity_numba(mask, pixels), dict)


@requires_numba
@pytest.mark.parametrize("legacy", [False, True], ids=["new", "legacy"])
@pytest.mark.parametrize("kind", ["stacked4d", "list"])
def test_numba_intensity_batch_matches_per_image(kind, legacy):
    """Batch (4D or list) == looping the single-image path; single = batch-of-1."""
    from cp_measure.core.numba.measureobjectintensity import (
        get_intensity as intensity_numba,
    )

    scenes = [_mask_pixels_2d() for _ in range(3)]
    if kind == "stacked4d":
        masks = np.stack([m[np.newaxis] for m, _ in scenes])  # (B, 1, H, W)
        pixels = np.stack([p[np.newaxis] for _, p in scenes])
    else:
        masks = [m for m, _ in scenes]
        pixels = [p for _, p in scenes]

    out = intensity_numba(masks, pixels, legacy=legacy)
    assert isinstance(out, list) and len(out) == 3
    for (m, p), got in zip(scenes, out):
        if kind == "stacked4d":
            m, p = m[np.newaxis], p[np.newaxis]
        _assert_dicts_match(intensity_numba(m, p, legacy=legacy), got)


@requires_numba
def test_numba_intensity_singleton_z_retains_3d_legacy_mad():
    from cp_measure.core.numba.measureobjectintensity import (
        get_intensity as intensity_numba,
    )

    mask = np.ones((2, 3), dtype=np.int32)
    pixels = np.arange(6, dtype=float).reshape(mask.shape)
    two_d = intensity_numba(mask, pixels, edge_measurements=False, legacy=True)
    three_d = intensity_numba(
        mask[np.newaxis], pixels[np.newaxis], edge_measurements=False, legacy=True
    )
    reference = intensity_numpy(
        mask[np.newaxis], pixels[np.newaxis], edge_measurements=False, legacy=True
    )
    _assert_dicts_match(reference, three_d)
    assert not np.array_equal(
        two_d["Intensity_MADIntensity"], three_d["Intensity_MADIntensity"]
    )


@requires_numba
def test_set_accelerator_numba_composes_with_numpy():
    cp_measure.set_accelerator("numba")
    try:
        core = cp_measure.bulk.get_core_measurements()
        assert core["intensity"].__module__ == (
            "cp_measure.core.numba.measureobjectintensity"
        )
        assert core["texture"].__module__ == "cp_measure.core.numba.measuretexture"
        assert core["feret"].__module__ == "cp_measure.core.numba._feret"
        assert core["granularity"].__module__ == (
            "cp_measure.core.numba.measuregranularity"
        )
        assert core["zernike"].__module__ == (
            "cp_measure.core.numba.measureobjectsizeshape"
        )
        assert core["radial_zernikes"].__module__ == (
            "cp_measure.core.numba.measureobjectintensitydistribution"
        )
        assert core["radial_distribution"].__module__ == (
            "cp_measure.core.numba.measureobjectintensitydistribution"
        )
        corr = cp_measure.bulk.get_correlation_measurements()
        assert list(corr) == ["costes", "pearson", "manders_fold", "rwc"]
        for feature in ("pearson", "manders_fold", "rwc", "costes"):
            assert corr[feature].__module__ == (
                "cp_measure.core.numba.measurecolocalization"
            )
        # Every other feature stays on the numpy backend.
        assert core["sizeshape"].__module__ == "cp_measure.core.measureobjectsizeshape"
    finally:
        cp_measure.set_accelerator(None)

    restored = cp_measure.bulk.get_core_measurements()
    assert restored["intensity"].__module__ == "cp_measure.core.measureobjectintensity"
    assert restored["feret"].__module__ == "cp_measure.core.measureobjectsizeshape"
    assert restored["granularity"].__module__ == "cp_measure.core.measuregranularity"


@requires_numba
@pytest.mark.parametrize(
    "feature,mask_name",
    [
        ("zernike", "masks"),
        ("radial_zernikes", "labels"),
        ("radial_distribution", "labels"),
    ],
)
def test_numba_accepts_numpy_keyword_names(feature, mask_name):
    """Switching the accelerator must not break callers using numpy's keywords."""
    masks, pixels = _mask_pixels_2d()
    cp_measure.set_accelerator("numba")
    try:
        numba_fn = cp_measure.bulk.get_core_measurements()[feature]
        assert numba_fn(**{mask_name: masks, "pixels": pixels})
    finally:
        cp_measure.set_accelerator(None)


def test_set_accelerator_numba_absent_raises(monkeypatch):
    """When find_spec reports numba absent, selecting it raises (no silent fallback)."""
    monkeypatch.setattr(cp_measure._detect, "HAS_NUMBA", False)
    cp_measure.set_accelerator("numba")
    try:
        with pytest.raises(RuntimeError, match="numba is not installed"):
            cp_measure.bulk.get_core_measurements()
    finally:
        cp_measure.set_accelerator(None)
