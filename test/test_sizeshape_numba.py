"""Numba size/shape geometry and current-main API parity regressions."""

import numpy
import pytest
import skimage.measure

from cp_measure._detect import HAS_NUMBA

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


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


def _assert_features_equal(got, ref):
    assert list(got) == list(ref)
    # Only perimeter-weight summation changes floating-point order. Moment,
    # orientation, Euler and schema parity must not be hidden by float casts.
    perimeter_features = {"Perimeter", "PerimeterCrofton", "FormFactor", "Compactness"}
    for key in ref:
        assert got[key].shape == ref[key].shape, key
        assert got[key].dtype == ref[key].dtype, key
        if key in perimeter_features:
            numpy.testing.assert_allclose(
                got[key],
                ref[key],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
                err_msg=key,
            )
        else:
            numpy.testing.assert_array_equal(got[key], ref[key], err_msg=key)


@requires_numba
@pytest.mark.parametrize("calculate_advanced", [False, True])
@pytest.mark.parametrize("new_features", [False, True])
@pytest.mark.parametrize("kind", ["empty", "ring", "degenerate", "irregular"])
def test_get_sizeshape_schema_and_numerics(kind, calculate_advanced, new_features):
    from cp_measure.core.measureobjectsizeshape import get_sizeshape as baseline
    from cp_measure.core.numba._sizeshape import get_sizeshape

    masks = _ring()
    if kind == "empty":
        masks[:] = 0
    elif kind == "degenerate":
        masks[:] = 0
        masks[2, 3] = 1
        masks[10:20, 15] = 2
    elif kind == "irregular":
        masks = numpy.random.default_rng(78).integers(0, 4, (40, 40), dtype=numpy.int32)
    pixels = numpy.random.default_rng(0).random(masks.shape)
    before, pixels_before = masks.copy(), pixels.copy()
    kwargs = dict(calculate_advanced=calculate_advanced, new_features=new_features)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        _assert_features_equal(
            get_sizeshape(masks, pixels, **kwargs), baseline(masks, pixels, **kwargs)
        )
    numpy.testing.assert_array_equal(masks, before)
    numpy.testing.assert_array_equal(pixels, pixels_before)


@requires_numba
@pytest.mark.parametrize(
    "dtype", [numpy.uint8, numpy.uint16, numpy.uint64, numpy.int32, numpy.int64, ">i4"]
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_get_sizeshape_dtype_and_layout(dtype, layout):
    from cp_measure.core.measureobjectsizeshape import get_sizeshape as baseline
    from cp_measure.core.numba._sizeshape import get_sizeshape

    masks = _ring().astype(dtype, order="F" if layout == "F" else "C")
    if layout == "strided":
        masks = masks[::-1, ::-1]
    masks.flags.writeable = False
    _assert_features_equal(get_sizeshape(masks, None), baseline(masks, None))


def _ellipses(size=1024, n=128):
    # Same seed and geometry as the earlier draft-ranking s1024_n128 case:
    # includes +/-90 orientation and cancellation-sensitive CentralMoment_2_3.
    rng = numpy.random.default_rng(20261020)
    yy, xx = numpy.mgrid[:size, :size]
    masks = numpy.zeros((size, size), numpy.int32)
    cols = int(numpy.ceil(numpy.sqrt(n)))
    rows = int(numpy.ceil(n / cols))
    for k in range(n):
        r, c = divmod(k, cols)
        cy = (r + 0.5) * size / rows + (0.5 if k % 3 == 0 else 0)
        cx = (c + 0.5) * size / cols + (0.5 if k % 3 == 0 else 0)
        a = (0.27 + 0.06 * rng.random()) * size / rows
        b = (0.27 + 0.06 * rng.random()) * size / cols
        masks[((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2 <= 1] = k + 1
    return masks


@requires_numba
def test_get_sizeshape_orientation_and_central_moment_regressions():
    from cp_measure.core.measureobjectsizeshape import get_sizeshape as baseline
    from cp_measure.core.numba._sizeshape import get_sizeshape

    masks = _ellipses()
    ref = baseline(masks, None)
    assert numpy.any(numpy.abs(ref["Orientation"]) == 90)
    _assert_features_equal(get_sizeshape(masks, None), ref)


@requires_numba
@pytest.mark.parametrize(
    "calculate_advanced,new_features", [(False, False), (True, True)]
)
def test_get_sizeshape_3d_fallback(calculate_advanced, new_features):
    from cp_measure.core.measureobjectsizeshape import get_sizeshape as baseline
    from cp_measure.core.numba._sizeshape import get_sizeshape

    masks = numpy.zeros((8, 20, 20), numpy.int32)
    masks[2:6, 3:14, 4:16] = 1
    pixels = numpy.ones_like(masks, dtype=float)
    kwargs = dict(
        calculate_advanced=calculate_advanced,
        new_features=new_features,
        spacing=(2.0, 1.0, 1.0),
    )
    _assert_features_equal(
        get_sizeshape(masks, pixels, **kwargs), baseline(masks, pixels, **kwargs)
    )


@requires_numba
@pytest.mark.parametrize("shape", [(1, 40, 40), (1, 1, 40, 40)])
def test_get_sizeshape_preserves_unsupported_rank_errors(shape):
    from cp_measure.core.measureobjectsizeshape import get_sizeshape as baseline
    from cp_measure.core.numba._sizeshape import get_sizeshape

    masks = _ring().reshape(shape)
    # Main cannot run marching_cubes on a singleton Z, nor regionprops on 4D.
    # Accelerating 2D must not silently turn either into a successful 2D call.
    with pytest.raises((ValueError, TypeError)) as ref:
        baseline(masks, None)
    with pytest.raises(type(ref.value)) as got:
        get_sizeshape(masks, None)
    assert str(got.value) == str(ref.value)


@requires_numba
def test_get_sizeshape_signature_and_pixel_shape_validation():
    import inspect
    from cp_measure.core.measureobjectsizeshape import get_sizeshape as baseline
    from cp_measure.core.numba._sizeshape import get_sizeshape

    assert inspect.signature(get_sizeshape) == inspect.signature(baseline)
    for fn in (baseline, get_sizeshape):
        with pytest.raises(ValueError, match="shape"):
            fn(_ring(), numpy.zeros((2, 2)))


@requires_numba
def test_get_sizeshape_bulk_and_featurizer_label_mapping():
    import cp_measure
    from cp_measure.bulk import get_core_measurements
    from cp_measure.featurizer import featurize, make_featurizer_config
    from conftest import ALL_OFF

    masks = _ring()
    masks[masks == 1] = 99
    masks[masks == 2] = 7
    before = masks.copy()
    pixels = numpy.ones(masks.shape)
    config = make_featurizer_config(["DNA"], **{**ALL_OFF, "sizeshape": True})
    previous = cp_measure._ACCELERATOR
    try:
        cp_measure.set_accelerator(None)
        ref = get_core_measurements()["sizeshape"](masks, pixels)
        ref_data, ref_columns, ref_rows = featurize(
            pixels[None, None], masks[None, None], config
        )
        cp_measure.set_accelerator("numba")
        got = get_core_measurements()["sizeshape"](masks, pixels)
        data, columns, rows = featurize(pixels[None, None], masks[None, None], config)
        _assert_features_equal(got, ref)
        assert columns == ref_columns
        assert rows == ref_rows
        assert [row[-1] for row in rows] == [7, 99]
        numpy.testing.assert_allclose(
            data, ref_data, rtol=1e-12, atol=1e-12, equal_nan=True
        )
    finally:
        cp_measure.set_accelerator(previous)
    numpy.testing.assert_array_equal(masks, before)
