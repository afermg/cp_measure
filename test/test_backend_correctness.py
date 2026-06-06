"""Backend correctness harness: numba `intensity` must match the numpy backend.

Compares ``cp_measure.core.numba.get_intensity`` against the reference
``cp_measure.core.get_intensity`` key-by-key on continuous random pixels (no
exact ties, so ``Location_MaxIntensity_*`` is unambiguous), for 2D and 3D input
with edge measurements on and off. Also checks the dispatch wiring: under
``set_accelerator("numba")`` the core registry composes the numba intensity with
the numpy implementations of every other feature, and an absent numba backend
raises rather than silently falling back.
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
@pytest.mark.parametrize("edge", [True, False], ids=["edge", "noedge"])
@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_numba_intensity_matches_numpy(dim, edge):
    from cp_measure.core.numba import get_intensity as intensity_numba

    mask, pixels = _mask_pixels_2d() if dim == "2d" else _mask_pixels_3d()
    ref = intensity_numpy(mask, pixels, edge_measurements=edge)
    got = intensity_numba(mask, pixels, edge_measurements=edge)
    _assert_dicts_match(ref, got)


@requires_numba
def test_set_accelerator_numba_composes_with_numpy():
    cp_measure.set_accelerator("numba")
    try:
        core = cp_measure.bulk.get_core_measurements()
        assert core["intensity"].__module__ == (
            "cp_measure.core.numba.measureobjectintensity"
        )
        # The colocalization features route to the numba backend too.
        corr = cp_measure.bulk.get_correlation_measurements()
        for feature in ("pearson", "manders_fold", "rwc", "costes", "overlap"):
            assert corr[feature].__module__ == (
                "cp_measure.core.numba.measurecolocalization"
            ), feature
        # Every other feature stays on the numpy backend.
        assert core["sizeshape"].__module__ == "cp_measure.core.measureobjectsizeshape"
        assert core["texture"].__module__ == "cp_measure.core.measuretexture"
    finally:
        cp_measure.set_accelerator(None)

    restored = cp_measure.bulk.get_core_measurements()
    assert restored["intensity"].__module__ == "cp_measure.core.measureobjectintensity"
    # overlap is numba-only; the default registry does not expose it.
    assert "overlap" not in cp_measure.bulk.get_correlation_measurements()


def test_set_accelerator_numba_absent_raises(monkeypatch):
    """When find_spec reports numba absent, selecting it raises (no silent fallback)."""
    monkeypatch.setattr(cp_measure._detect, "HAS_NUMBA", False)
    cp_measure.set_accelerator("numba")
    try:
        with pytest.raises(RuntimeError, match="numba is not installed"):
            cp_measure.bulk.get_core_measurements()
    finally:
        cp_measure.set_accelerator(None)
