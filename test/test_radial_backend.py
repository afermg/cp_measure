"""Backend correctness for numba radial_distribution.

This lane intentionally FIXES Issue #22 (per-object results must not depend on
other labels), so it does NOT match the buggy numpy baseline on multi-object
fields. Exactness is anchored instead on the correct reference: the numba result
for object k in a multi-object field must equal the numpy baseline run on object k
in ISOLATION. Plus an independence test (the #22 property itself), a direct
single-object golden, and the 2D-only / batch bzyx paths.
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
    """The Issue #22 example: two objects with edge asymmetries (EVEN-sized
    squares → a symmetric 2×2 centre plateau). Used for the field-independence
    test, which holds regardless of centre ties."""
    masks = np.zeros((240, 240), np.int32)
    masks[50:100, 50:100] = 1
    masks[80:120, 90:120] = 1
    masks[150:200, 150:200] = 2
    masks[175:180, 180:210] = 2
    rng = np.random.default_rng(42)
    pixels = rng.random((240, 240))
    return masks, pixels


def _unique_centre_masks():
    """Two objects with a UNIQUE farthest-from-edge pixel (ODD-sized squares).

    The exact golden vs numpy needs an unambiguous centre: when an object has a
    symmetric centre *plateau* (even-sized), the reference's
    ``scipy.ndimage.maximum_position`` tie-break is field/layout-dependent and a
    per-object crop legitimately picks a different (equally-valid) centre — the
    numba centre is deterministic and field-independent by design. Odd-sized
    objects have a single centre pixel, so both agree to machine precision.
    """
    masks = np.zeros((240, 240), np.int32)
    masks[50:101, 50:101] = 1
    masks[80:121, 90:121] = 1
    masks[150:201, 150:201] = 2
    masks[175:180, 180:211] = 2
    rng = np.random.default_rng(42)
    pixels = rng.random((240, 240))
    return masks, pixels


def _assert_close(a, b, key=""):
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=key)


@requires_numba
@pytest.mark.parametrize("scaled", [True, False])
def test_numba_multi_equals_numpy_isolated(scaled):
    """numba(multi)[k] == numpy baseline on object k ALONE (the correct reference)."""
    masks, pixels = _unique_centre_masks()
    got = _numba()(masks, pixels, scaled=scaled)
    for k, label in enumerate((1, 2)):
        iso = (masks == label).astype(np.int32)  # object alone, relabelled to 1
        exp = ref(iso, pixels, scaled=scaled)
        assert set(got) == set(exp)
        for key in exp:
            _assert_close(got[key][k], exp[key][0], key)


@requires_numba
def test_per_object_independence():
    """The Issue #22 property: an object's numba result is field-independent."""
    masks, pixels = _issue22_masks()
    got = _numba()(masks, pixels)
    for k, label in enumerate((1, 2)):
        alone = _numba()((masks == label).astype(np.int32), pixels)
        for key in got:
            _assert_close(got[key][k], alone[key][0], key)


@requires_numba
def test_single_object_matches_numpy_directly():
    """One (unique-centre) object: baseline is correct, so numba == numpy directly."""
    masks, pixels = _unique_centre_masks()
    masks = (masks == 1).astype(np.int32)
    _assert_close_dict(_numba()(masks, pixels), ref(masks, pixels))


def _assert_close_dict(got, exp):
    assert set(got) == set(exp), set(got).symmetric_difference(exp)
    for key in exp:
        _assert_close(got[key], exp[key], key)


@requires_numba
def test_3d_volume_returns_empty():
    vol = np.zeros((3, 32, 32), np.int32)
    vol[:, 5:15, 5:15] = 1
    pix = np.random.default_rng(0).random((3, 32, 32))
    assert _numba()(vol, pix) == {}


@requires_numba
def test_empty_image_empty_arrays():
    masks = np.zeros((40, 40), np.int32)
    pix = np.random.default_rng(0).random((40, 40))
    got = _numba()(masks, pix)
    assert all(v.shape == (0,) for v in got.values())
    assert set(got) == set(ref(masks, pix))


@requires_numba
def test_batch_list_matches_per_image():
    m0, p0 = _issue22_masks()
    m1 = (m0 == 1).astype(np.int32)
    got = _numba()([m0, m1], [p0, p0])
    assert isinstance(got, list) and len(got) == 2
    for per_image, m in zip(got, (m0, m1)):
        _assert_close_dict(per_image, _numba()(m, p0))
