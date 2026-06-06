"""Golden tests: numba spatial-moment kernel must match the numpy accumulator and regionprops.

Phase 1 of the unified numba sizeshape lane. The fused numba kernel computes the raw + central
moment matrices; the derived quantities (normalized / Hu / inertia) reuse the shared algebra in
``cp_measure.primitives._moments``. Raw moments are bit-exact; centroid-dependent matrices match
to floating-point round-off.
"""

import numpy
import pytest
import skimage.measure

from cp_measure._detect import HAS_NUMBA
from cp_measure.primitives import _moments as M

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")

ATOL_REL = 1e-7


def _square_objects(size, n, gap_frac=0.7):
    masks = numpy.zeros((size, size), numpy.int32)
    step = size // n
    obj = int(step * gap_frac)
    lab = 0
    for a in range(n):
        for b in range(n):
            lab += 1
            masks[a * step : a * step + obj, b * step : b * step + obj] = lab
    return masks


def _assert_numba_matches_numpy(masks):
    from cp_measure.core.numba._sizeshape import spatial_moments_2d as numba_moments

    rawN, cenN, normN, huN = numba_moments(masks)
    rawP, cenP, normP, huP = M.spatial_moments_2d(masks)
    numpy.testing.assert_array_equal(rawN, rawP)  # raw is bit-exact
    numpy.testing.assert_array_equal(cenN, cenP)  # same kernel order -> bit-identical
    numpy.testing.assert_array_equal(huN, huP)
    assert numpy.array_equal(numpy.isnan(normN), numpy.isnan(normP))
    numpy.testing.assert_array_equal(
        normN[~numpy.isnan(normN)], normP[~numpy.isnan(normP)]
    )


def _assert_numba_matches_regionprops(masks):
    from cp_measure.core.numba._sizeshape import spatial_moments_2d as numba_moments

    raw, central, normalized, hu = numba_moments(masks)
    ref = skimage.measure.regionprops_table(
        masks,
        properties=["moments", "moments_central", "moments_normalized", "moments_hu"],
    )
    n = raw.shape[0]
    for p in range(4):
        for q in range(4):
            numpy.testing.assert_array_equal(raw[:, p, q], ref[f"moments-{p}-{q}"])
            s = max(
                numpy.nanmax(numpy.abs(ref[f"moments_central-{p}-{q}"])) if n else 0.0,
                1.0,
            )
            numpy.testing.assert_allclose(
                central[:, p, q],
                ref[f"moments_central-{p}-{q}"],
                rtol=ATOL_REL,
                atol=ATOL_REL * s,
            )
            numpy.testing.assert_allclose(
                normalized[:, p, q],
                ref[f"moments_normalized-{p}-{q}"],
                rtol=ATOL_REL,
                atol=ATOL_REL,
                equal_nan=True,
            )
    for k in range(7):
        s = max(numpy.nanmax(numpy.abs(ref[f"moments_hu-{k}"])) if n else 0.0, 1.0)
        numpy.testing.assert_allclose(
            hu[:, k], ref[f"moments_hu-{k}"], rtol=ATOL_REL, atol=ATOL_REL * s
        )


@requires_numba
def test_numba_moments_match_numpy_multi():
    _assert_numba_matches_numpy(_square_objects(256, 4))


@requires_numba
def test_numba_moments_match_regionprops_multi():
    _assert_numba_matches_regionprops(_square_objects(256, 4))


@requires_numba
def test_numba_moments_noncontiguous_labels():
    masks = numpy.zeros((96, 96), numpy.int32)
    masks[10:30, 10:30] = 1
    masks[40:60, 40:60] = 3
    masks[70:90, 70:90] = 7
    _assert_numba_matches_numpy(masks)
    _assert_numba_matches_regionprops(masks)


@requires_numba
def test_numba_moments_edge_touching():
    masks = numpy.zeros((64, 64), numpy.int32)
    masks[0:20, 0:20] = 1
    masks[44:64, 44:64] = 2
    _assert_numba_matches_regionprops(masks)


@requires_numba
def test_numba_moments_single_pixel():
    masks = numpy.zeros((32, 32), numpy.int32)
    masks[16, 16] = 1
    masks[5:15, 5:15] = 2
    _assert_numba_matches_numpy(masks)


@requires_numba
def test_numba_moments_inertia_matches_regionprops():
    from cp_measure.core.numba._sizeshape import spatial_moments_2d as numba_moments

    masks = _square_objects(200, 3)
    _, central, _, _ = numba_moments(masks)
    it_00, it_off, it_11, eig_0, eig_1 = M.inertia_2d(central)
    ref = skimage.measure.regionprops_table(
        masks, properties=["inertia_tensor", "inertia_tensor_eigvals"]
    )
    for got, key in [
        (it_00, "inertia_tensor-0-0"),
        (it_off, "inertia_tensor-0-1"),
        (it_11, "inertia_tensor-1-1"),
        (eig_0, "inertia_tensor_eigvals-0"),
        (eig_1, "inertia_tensor_eigvals-1"),
    ]:
        s = max(numpy.nanmax(numpy.abs(ref[key])), 1.0)
        numpy.testing.assert_allclose(got, ref[key], rtol=ATOL_REL, atol=ATOL_REL * s)


@requires_numba
def test_numba_moments_empty():
    from cp_measure.core.numba._sizeshape import spatial_moments_2d as numba_moments

    raw, central, normalized, hu = numba_moments(numpy.zeros((20, 20), numpy.int32))
    assert raw.shape == (0, 4, 4) and hu.shape == (0, 7)


# --- convex hull (area_convex) ---


def _assert_convex_matches(masks):
    from cp_measure.core.numba._sizeshape import convex_area_2d

    got = convex_area_2d(masks)
    ref = skimage.measure.regionprops_table(masks, properties=["area_convex"])[
        "area_convex"
    ]
    numpy.testing.assert_array_equal(got, ref)  # rasterised pixel count -> bit-exact


@requires_numba
def test_numba_convex_area_multi():
    _assert_convex_matches(_square_objects(256, 4))


@requires_numba
def test_numba_convex_area_irregular():
    rng = numpy.random.default_rng(0)
    masks = numpy.zeros((128, 128), numpy.int32)
    yy, xx = numpy.mgrid[0:128, 0:128]
    for lab, (cy, cx) in enumerate(rng.integers(20, 108, size=(6, 2)), 1):
        masks[(yy - cy) ** 2 + (xx - cx) ** 2 < rng.integers(40, 160)] = lab
    _assert_convex_matches(masks)


@requires_numba
def test_numba_convex_area_noncontiguous():
    masks = numpy.zeros((96, 96), numpy.int32)
    masks[10:30, 10:30] = 1
    masks[40:60, 40:60] = 3
    masks[70:90, 70:90] = 7
    _assert_convex_matches(masks)


@requires_numba
def test_numba_convex_area_edge_touching():
    masks = numpy.zeros((64, 64), numpy.int32)
    masks[0:20, 0:20] = 1
    masks[44:64, 44:64] = 2
    _assert_convex_matches(masks)


@requires_numba
def test_numba_convex_area_degenerate():
    # single pixel and a 1-wide line: hull is the pixels themselves (area == pixel count).
    masks = numpy.zeros((32, 32), numpy.int32)
    masks[16, 16] = 1
    masks[5, 5:15] = 2
    masks[20:28, 20:24] = 3
    _assert_convex_matches(masks)


@requires_numba
def test_numba_convex_area_empty():
    from cp_measure.core.numba._sizeshape import convex_area_2d

    assert convex_area_2d(numpy.zeros((20, 20), numpy.int32)).shape == (0,)


# --- perimeter / perimeter_crofton / euler_number ---


def _ring(size=40):
    """An object with a hole (euler_number = 0) plus a solid one (euler_number = 1)."""
    m = numpy.zeros((size, size), numpy.int32)
    m[5:25, 5:25] = 1
    m[10:20, 10:20] = 0  # punch a hole -> object 1 has euler 0
    m[28:36, 28:36] = 2
    return m


def _assert_per_crofton_euler(masks):
    from cp_measure.core.numba._sizeshape import crofton_euler_2d, perimeter_2d

    ref = skimage.measure.regionprops_table(
        masks, properties=["perimeter", "perimeter_crofton", "euler_number"]
    )
    per = perimeter_2d(masks)
    cro, eul = crofton_euler_2d(masks)
    numpy.testing.assert_allclose(per, ref["perimeter"], rtol=1e-9, atol=1e-9)
    numpy.testing.assert_allclose(cro, ref["perimeter_crofton"], rtol=1e-9, atol=1e-9)
    numpy.testing.assert_array_equal(eul, ref["euler_number"])  # integer -> exact


@requires_numba
def test_numba_perimeter_euler_multi():
    _assert_per_crofton_euler(_square_objects(256, 4))


@requires_numba
def test_numba_perimeter_euler_holed():
    _assert_per_crofton_euler(_ring())


@requires_numba
def test_numba_perimeter_euler_noncontiguous():
    masks = numpy.zeros((96, 96), numpy.int32)
    masks[10:30, 10:30] = 1
    masks[40:60, 40:60] = 3
    masks[70:90, 70:90] = 7
    _assert_per_crofton_euler(masks)


@requires_numba
def test_numba_perimeter_euler_edge_touching():
    masks = numpy.zeros((64, 64), numpy.int32)
    masks[0:20, 0:20] = 1
    masks[44:64, 44:64] = 2
    _assert_per_crofton_euler(masks)


@requires_numba
def test_numba_perimeter_euler_irregular():
    rng = numpy.random.default_rng(1)
    masks = numpy.zeros((128, 128), numpy.int32)
    yy, xx = numpy.mgrid[0:128, 0:128]
    for lab, (cy, cx) in enumerate(rng.integers(20, 108, size=(6, 2)), 1):
        masks[(yy - cy) ** 2 + (xx - cx) ** 2 < rng.integers(40, 160)] = lab
    _assert_per_crofton_euler(masks)


@requires_numba
def test_numba_perimeter_euler_empty():
    from cp_measure.core.numba._sizeshape import crofton_euler_2d, perimeter_2d

    assert perimeter_2d(numpy.zeros((20, 20), numpy.int32)).shape == (0,)
    cro, eul = crofton_euler_2d(numpy.zeros((20, 20), numpy.int32))
    assert cro.shape == (0,) and eul.shape == (0,)


# --- end-to-end get_sizeshape wrapper ---


def _assert_full_sizeshape_matches(masks, pixels, **kw):
    import cp_measure.core.measureobjectsizeshape as numpy_ss
    from cp_measure.core.numba._sizeshape import get_sizeshape as numba_ss

    got = numba_ss(masks, pixels, **kw)
    ref = numpy_ss.get_sizeshape(masks, pixels, **kw)
    assert set(got) == set(ref), set(got).symmetric_difference(ref)
    for k in ref:
        g, r = numpy.asarray(got[k], float), numpy.asarray(ref[k], float)
        assert g.shape == r.shape, k
        finite = numpy.abs(r[numpy.isfinite(r)])  # NormalizedMoment_0_* are always NaN
        s = max(finite.max() if finite.size else 0.0, 1.0)
        numpy.testing.assert_allclose(
            g, r, rtol=1e-7, atol=1e-7 * s, equal_nan=True, err_msg=k
        )


@requires_numba
def test_get_sizeshape_matches_numpy_multi():
    masks = _square_objects(256, 4)
    _assert_full_sizeshape_matches(
        masks, _pixels := numpy.random.default_rng(0).random(masks.shape)
    )


@requires_numba
def test_get_sizeshape_matches_numpy_noncontiguous():
    masks = numpy.zeros((96, 96), numpy.int32)
    masks[10:30, 10:30] = 1
    masks[40:60, 40:60] = 3
    masks[70:90, 70:90] = 7
    _assert_full_sizeshape_matches(
        masks, numpy.random.default_rng(1).random(masks.shape)
    )


@requires_numba
def test_get_sizeshape_flag_variants():
    masks = _square_objects(160, 3)
    pixels = numpy.random.default_rng(2).random(masks.shape)
    _assert_full_sizeshape_matches(masks, pixels, calculate_advanced=False)
    _assert_full_sizeshape_matches(masks, pixels, new_features=False)
    _assert_full_sizeshape_matches(
        masks, pixels, calculate_advanced=False, new_features=False
    )


@requires_numba
def test_get_sizeshape_3d_falls_back_to_numpy():
    import cp_measure.core.measureobjectsizeshape as numpy_ss
    from cp_measure.core.numba._sizeshape import get_sizeshape as numba_ss

    rng = numpy.random.default_rng(3)
    masks = numpy.zeros((12, 32, 32), numpy.int32)
    zz, yy, xx = numpy.mgrid[0:12, 0:32, 0:32]
    masks[(zz - 6) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2 < 60] = 1
    pixels = rng.random((12, 32, 32))
    got = numba_ss(masks, pixels)
    ref = numpy_ss.get_sizeshape(masks, pixels)
    assert set(got) == set(ref)
    for k in ref:
        numpy.testing.assert_allclose(
            numpy.asarray(got[k], float),
            numpy.asarray(ref[k], float),
            rtol=1e-7,
            atol=1e-6,
            equal_nan=True,
        )


@requires_numba
def test_get_sizeshape_dispatch():
    import cp_measure
    from cp_measure.bulk import _dispatch
    from cp_measure.core.numba._sizeshape import get_sizeshape as numba_ss

    cp_measure.set_accelerator("numba")
    try:
        assert _dispatch("core")["sizeshape"] is numba_ss
    finally:
        cp_measure.set_accelerator(None)
