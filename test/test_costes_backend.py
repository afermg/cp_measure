"""Current-main Costes API, undefined-result and batch contracts for Numba."""

import inspect
import warnings

import numpy as np
import pytest

import cp_measure
from cp_measure import bulk
from cp_measure._detect import HAS_NUMBA
from cp_measure.core import measurecolocalization as ref
from test_costes import FIXTURES, arrays

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
MODES = (ref.M_FASTER, ref.M_FAST, ref.M_ACCURATE)


def parity(first, second, masks, **kwargs):
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    original = [x.copy() for x in (first, second, masks)]
    results, diagnostics = [], []
    for function in (ref.get_correlation_costes, get_correlation_costes):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            results.append(function(first, second, masks, **kwargs))
        diagnostics.append([(w.category, str(w.message)) for w in recorded])
    expected, actual = results
    assert list(actual) == list(expected)
    assert diagnostics[0] == diagnostics[1]
    for key in expected:
        assert isinstance(actual[key], list)
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
        assert np.asarray(actual[key]).dtype == np.asarray(expected[key]).dtype
    for before, after in zip(original, (first, second, masks)):
        np.testing.assert_array_equal(before, after)
    return actual


@pytest.mark.parametrize("case", FIXTURES)
@pytest.mark.parametrize(
    "encoding", ["integer", "division", "float32", "signed", "mixed"]
)
def test_costes_production_scale_and_fixtures(case, encoding):
    a, b = arrays(case)
    scale = case["scale_max"]
    kwargs = {"fast_costes": case["mode"]}
    if encoding != "integer":
        kwargs["scale_max"] = scale
    if encoding in ("division", "float32"):
        dtype = np.float32 if encoding == "float32" else np.float64
        a, b = (a / scale).astype(dtype), (b / scale).astype(dtype)
    elif encoding == "signed":
        a, b = a.astype(np.int32), b.astype(np.int32)
    elif encoding == "mixed":
        b = b / scale
    parity(a, b, np.ones(a.shape, np.int32), **kwargs)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "case",
    [
        "zero",
        "constant",
        "tiny",
        "nan",
        "inf",
        "covariance",
        "anticorrelated",
        "gaps",
        "empty",
        "constant_background",
    ],
)
def test_costes_undefined_warnings_and_labels(mode, case):
    a = np.array([[0.1, 0.2, 0.4, 0.7], [0.3, 0.5, 0.8, 1.0]])
    b = a[:, ::-1].copy()
    masks = np.ones(a.shape, dtype=np.int32)
    if case == "zero":
        a[:] = b[:] = 0
    elif case == "constant":
        a[:] = 0.1
    elif case == "tiny":
        masks[:] = 0
        masks[0, :2] = 1
    elif case in ("nan", "inf"):
        a[0, 1] = np.nan if case == "nan" else np.inf
    elif case == "covariance":
        a = np.array([[0.1, 0.1, 0.9, 0.9]])
        b = np.array([[0.1, 0.9, 0.1, 0.9]])
        masks = np.ones(a.shape, dtype=int)
    elif case == "anticorrelated":
        b = 1 - a
    elif case == "gaps":
        masks[1] = 3
    elif case == "empty":
        masks[:] = 0
    elif case == "constant_background":
        a = np.r_[np.zeros(12), np.arange(1, 7)].reshape(3, 6) / 10
        b = np.r_[np.zeros(12), [1, 3, 2, 5, 4, 6]].reshape(3, 6) / 10
        masks = np.ones(a.shape, dtype=int)
    parity(a, b, masks, scale_max=255, fast_costes=mode)


def mixed_status_objects():
    """One object per row: defined, undefined, resumed and replayed objects, and
    a missing label, interleaved so the order of warnings and replays shows."""
    rng = np.random.default_rng(13)
    tie = (
        np.array([2, 2, 4, 4, 0, 7, 5, 6, 0, 5]) / 255,
        np.array([4, 2, 6, 3, 1, 6, 8, 8, 8, 0]) / 255,
    )
    near_constant = 0.1 + rng.random((2, 10)) * 1e-14
    rows = [
        (np.linspace(0.1, 0.9, 10), np.linspace(0.2, 0.8, 10) ** 2),  # defined
        (np.full(10, 0.1), np.linspace(0.1, 0.9, 10)),  # constant channel
        tie,  # a tie: resumed (Faster) or replayed with scipy_ties
        (np.zeros(10), np.zeros(10)),  # fewer than three nonzero pixels
        tuple(near_constant),  # near-constant: replayed with SciPy's warnings
        (np.full(10, 0.1), np.linspace(0.1, 0.9, 10)),  # constant again
    ]
    a = np.array([r[0] for r in rows])
    b = np.array([r[1] for r in rows])
    # label 4 is missing
    masks = np.repeat(np.array([1, 2, 3, 5, 6, 7]), 10).reshape(a.shape)
    return a, b, masks


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scipy_ties", [False, True])
def test_costes_mixed_statuses_keep_reference_order(mode, scipy_ties):
    """Values, warnings and their order match the reference across objects whose
    statuses interleave, including SciPy's warnings from replayed objects."""
    a, b, masks = mixed_status_objects()
    if not scipy_ties:
        # The kernel decides ties itself: drop the tie and near-constant objects.
        masks = np.where(np.isin(masks, [3, 6]), 0, masks)
    parity(a, b, masks, scale_max=255, fast_costes=mode, scipy_ties=scipy_ties)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_undefined_costes_warnings_are_summarized(backend):
    """One RuntimeWarning per undefined reason and image, with the number of objects,
    in the order the reasons first occur. A batch warns for each image."""
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    function = (
        get_correlation_costes if backend == "numba" else ref.get_correlation_costes
    )
    a, b, masks = mixed_status_objects()
    # labels 1, 2, 5 and 7 remain: defined, constant, all zero, constant
    masks = np.where(np.isin(masks, [3, 6]), 0, masks)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = function([a, a], [b, b], [masks, masks], scale_max=255)
    image = [
        "Undefined Costes result for 2 objects: constant channel in the regression subset",
        "Undefined Costes result for 3 objects: missing or empty object label",
        "Undefined Costes result for 1 object: fewer than three nonzero paired pixels",
    ]
    assert [(w.category, str(w.message)) for w in recorded] == [
        (RuntimeWarning, message) for message in image * 2
    ]
    for values in result:
        assert np.isfinite(values["Correlation_Costes_1"][0])
        assert np.isnan(values["Correlation_Costes_1"][1:]).all()


@pytest.mark.parametrize("mode", MODES)
def test_costes_out_of_range_after_warnings_matches_reference(mode):
    """An out-of-range object raises after the objects before it were measured,
    with the same warnings emitted first by both backends: SciPy's from replayed
    objects, but no undefined-result summary, as the call does not finish."""
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    a, b, masks = mixed_status_objects()
    a[5, 3] = 1.5  # the last object
    runs = []
    for function, extra in (
        (ref.get_correlation_costes, {}),
        (get_correlation_costes, {"scipy_ties": True}),
    ):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="normalized pixels"):
                function(a, b, masks, scale_max=255, fast_costes=mode, **extra)
        runs.append([(w.category, str(w.message)) for w in recorded])
    assert runs[0] == runs[1]
    assert runs[0] and not [m for _, m in runs[0] if "Undefined Costes" in m]


@pytest.mark.parametrize("scale", [None, 0, 1, -1, True, 255.0, np.nan, np.inf, 2**32])
def test_costes_invalid_grid(scale):
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    a = np.arange(8).reshape(2, 4) / 8
    with pytest.raises(ValueError, match="scale_max"):
        get_correlation_costes(a, a, np.ones(a.shape, int), scale_max=scale)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_costes_invalid_range(bad):
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    a = np.array([[bad, 0.1, 0.3, 0.7]])
    with pytest.raises(ValueError, match="normalized pixels"):
        get_correlation_costes(a, a, np.ones(a.shape, int), scale_max=255)


def test_costes_signature_and_validation():
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    assert inspect.signature(get_correlation_costes) == inspect.signature(
        ref.get_correlation_costes
    )
    a, b = arrays(FIXTURES[0])
    with pytest.raises(ValueError, match="mode"):
        get_correlation_costes(a, b, np.ones(a.shape, int), fast_costes="typo")
    with pytest.raises(ValueError, match="integer"):
        get_correlation_costes(a, b, np.ones(a.shape, bool))
    with pytest.raises(ValueError, match="same shape"):
        get_correlation_costes(a, b[None], np.ones(a.shape, int))
    with pytest.raises(ValueError, match="matching"):
        get_correlation_costes(a[None], b[None], np.ones(a.shape, int))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ["2d", "singleton", "volume", "strided"])
def test_costes_spatial_inputs(mode, shape):
    a, b = arrays(FIXTURES[0])
    if shape == "singleton":
        a, b = a[None], b[None]
    elif shape == "volume":
        a, b = np.stack([a, a]), np.stack([b, b])
    elif shape == "strided":
        a, b = a[:, ::-2], b[:, ::-2]
    parity(a, b, np.ones(a.shape, np.uint64), fast_costes=mode)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("encoding", ["integer", "division", "multiply"])
def test_costes_normalization_encodings_match_both_backends(mode, encoding):
    # Exact grid encodings must agree with integers, not merely within a backend.
    a = np.array(
        [43, 85, 88, 36, 90, 36, 61, 81, 89, 88, 79, 79, 48, 54], np.uint8
    ).reshape(2, 7)
    b = np.array([8, 8, 8, 8, 9, 9, 8, 8, 9, 9, 9, 9, 9, 10], np.uint8).reshape(2, 7)
    expected = ref.get_correlation_costes(a, b, np.ones(a.shape, int), fast_costes=mode)
    if encoding == "division":
        a, b = a / 255, b / 255
    elif encoding == "multiply":
        a, b = a * (1.0 / 255), b * (1.0 / 255)
    a.setflags(write=False)
    b.setflags(write=False)
    actual = parity(a, b, np.ones(a.shape, int), fast_costes=mode, scale_max=255)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("is_3d", [False, True])
def test_costes_bulk_and_featurizer_forwarding(is_3d):
    from cp_measure.featurizer import featurize, make_featurizer_config
    from conftest import ALL_OFF

    a, b = arrays(FIXTURES[0])
    image = np.stack([a, b]).astype(float) / 255
    masks = np.full((1, *a.shape), 42, dtype=int)
    if is_3d:
        image = np.stack([image, image], axis=1)
        masks = np.stack([masks, masks], axis=1)
    images = np.stack([image, image])
    labels = np.stack([masks, masks])
    labels[1] = 71
    labels[1, ..., : a.shape[-1] // 2] = 99
    config = make_featurizer_config(
        ["a", "b"],
        **{**ALL_OFF, "correlation_costes": True},
        correlation_costes_params={
            "scale_max": 255,
            "fast_costes": ref.M_FAST,
            "thr": 42,
        },
    )
    try:
        cp_measure.set_accelerator(None)
        expected = featurize(images, labels, config, is_3d=is_3d)
        cp_measure.set_accelerator("numba")
        assert (
            "numba"
            in bulk.get_correlation_measurements(sanitize=False)["costes"].__module__
        )
        actual = featurize(images, labels, config, is_3d=is_3d)
        assert actual[1:] == expected[1:]
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    finally:
        cp_measure.set_accelerator(None)


@pytest.mark.parametrize("mode", MODES)
def test_costes_zero_correlation_branch_replays_reference(mode, monkeypatch):
    # Discrete seed 96: scalar Pearson sums crossed zero in the wrong direction,
    # changing C1 by 0.042 and C2 by 0.118 before reference replay was added.
    a = (
        np.array(
            [
                5,
                4,
                7,
                6,
                0,
                2,
                6,
                5,
                5,
                1,
                5,
                3,
                1,
                6,
                5,
                4,
                5,
                6,
                0,
                3,
                5,
                4,
                1,
                2,
                7,
                5,
                5,
                5,
                2,
                3,
                7,
                2,
                4,
                1,
                3,
                1,
                0,
                7,
                7,
                6,
                6,
                2,
                6,
                4,
                4,
                0,
                0,
                7,
                2,
                5,
                0,
                7,
                4,
                0,
                0,
                7,
                6,
                1,
                7,
                7,
                6,
                4,
                5,
                7,
                1,
                3,
            ],
            dtype=float,
        ).reshape(1, -1)
        / 8
    )
    b = (
        np.array(
            [
                5,
                5,
                5,
                4,
                2,
                4,
                2,
                6,
                2,
                4,
                0,
                3,
                5,
                4,
                3,
                4,
                7,
                5,
                5,
                1,
                3,
                1,
                7,
                6,
                4,
                1,
                4,
                5,
                1,
                5,
                2,
                7,
                0,
                3,
                0,
                2,
                0,
                3,
                0,
                5,
                2,
                3,
                1,
                3,
                2,
                1,
                1,
                7,
                0,
                2,
                0,
                4,
                5,
                2,
                2,
                1,
                3,
                4,
                2,
                5,
                0,
                7,
                6,
                4,
                5,
                2,
            ],
            dtype=float,
        ).reshape(1, -1)
        / 8
    )
    parity(
        a, b, np.ones(a.shape, int), scale_max=255, fast_costes=mode, scipy_ties=True
    )
    # The reference decides the ambiguous zero crossing: by replaying the object,
    # or (Faster) by deciding only that step and resuming the compiled search.
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    decided = []
    for name in ("_costes_pair", "_costes_background_correlation"):
        original = getattr(ref, name)
        recording = lambda *args, _f=original, _n=name: decided.append(_n) or _f(*args)  # noqa: E731
        monkeypatch.setattr(ref, name, recording)
    get_correlation_costes(
        a, b, np.ones(a.shape, int), scale_max=255, fast_costes=mode, scipy_ties=True
    )
    assert decided


@pytest.mark.parametrize("mode", MODES)
def test_costes_near_constant_scipy_diagnostics(mode):
    rng = np.random.default_rng(13)
    a = 0.1 + rng.random((2, 12)) * 1e-14
    b = 0.1 + rng.random((2, 12)) * 1e-14
    parity(
        a, b, np.ones(a.shape, int), scale_max=255, fast_costes=mode, scipy_ties=True
    )


@pytest.mark.parametrize("scale", [2, 4095, 65535, 2**32 - 1])
def test_costes_explicit_acquisition_grids(scale):
    a, b = arrays(FIXTURES[0])
    parity(a / 255, b / 255, np.ones(a.shape, int), scale_max=scale)


def test_costes_background_ignored_and_unused_threshold():
    a, b = arrays(FIXTURES[0])
    a = np.pad(a / 255, 1, constant_values=np.nan)
    b = np.pad(b / 255, 1, constant_values=999)
    masks = np.pad(np.ones((8, 16), int), 1)
    parity(a, b, masks, scale_max=255, thr=93)


@pytest.mark.parametrize("name", ["get_correlation_costes", "get_correlation_rwc"])
def test_threaded_batch_matches_per_image_calls(name):
    """A batch runs on threads; values and warnings equal per-image calls, in order,
    including images whose objects warn or fall back to the reference."""
    from cp_measure.core.numba import measurecolocalization as numba_coloc

    function = getattr(numba_coloc, name)
    kwargs = {"scale_max": 255} if name == "get_correlation_costes" else {}
    a = np.array([[0.1, 0.2, 0.4, 0.7], [0.3, 0.5, 0.8, 1.0]] * 4)
    masks = np.repeat(np.arange(1, 5), 2)[:, None] * np.ones((1, 4), dtype=np.int32)
    constant, with_nan = a.copy(), a.copy()
    constant[:2] = 0.1
    with_nan[3, 1] = np.nan
    firsts = [a, constant, a, with_nan, a]
    seconds = [a[:, ::-1].copy(), a[:, ::-1].copy(), 1 - a, a[::-1].copy(), a**2]

    def run(call):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            values = call()
        return values, [(w.category, str(w.message)) for w in recorded]

    expected = [
        run(lambda f=f, s=s: function(f, s, masks, **kwargs))
        for f, s in zip(firsts, seconds)
    ]
    got, got_warnings = run(lambda: function(firsts, seconds, [masks] * 5, **kwargs))
    assert got_warnings == [w for _, ws in expected for w in ws]
    for item, (values, _) in zip(got, expected):
        assert list(item) == list(values)
        for key in values:
            np.testing.assert_array_equal(item[key], values[key])


@pytest.mark.parametrize("mode", MODES)
def test_costes_near_constant_float_object_matches_reference(mode):
    """|mean|/std ~ 1e10: the one-pass search steps must leave every decision the
    exact kernel would replay to the reference, so values and warnings match."""
    x = np.array([0.5001220702273792, 0.5001220702273793, 0.5001220702935895,
                  0.5001220703935095, 0.5001220703053191, 0.5001220702452209,
                  0.5001220703047751, 0.5001220702836826, 0.500122070423178,
                  0.5001220703200201])  # fmt: skip
    y = np.array([0.500122070337724, 0.5001220702661621, 0.5001220702523903,
                  0.5001220703152156, 0.5001220702994836, 0.5001220703285519,
                  0.5001220703045545, 0.5001220703443997, 0.5001220703058452,
                  0.5001220703133079])  # fmt: skip
    parity(
        x[None],
        y[None],
        np.ones((1, 10), int),
        scale_max=65535,
        fast_costes=mode,
        scipy_ties=True,
    )


def test_costes_resumed_bisection_matches_reference():
    """r is exactly 0 at a bisection step of this 8-bit object, so SciPy's last bits
    decide it: the kernel pauses, the reference decides that step, the kernel
    resumes, and values and warnings equal the reference."""
    from cp_measure.core.numba._costes import RESUME, costes_faster
    from cp_measure.primitives._segment_numba import numpy_sum_chunk

    a = np.array([[2, 2, 4, 4, 0, 7, 5, 6, 0, 5]]) / 255
    b = np.array([[4, 2, 6, 3, 1, 6, 8, 8, 8, 0]]) / 255
    status = costes_faster(
        a[0].copy(), b[0].copy(), np.array([0, 10]), 255.0, True, numpy_sum_chunk()
    )[2]
    assert status[0] == RESUME
    parity(a, b, np.ones(a.shape, int), scale_max=255, scipy_ties=True)


def test_costes_kernel_decides_ties_by_default():
    """By default the kernel decides an exact-zero tie and a near-constant subset
    itself: no SciPy call, no SciPy warning, and the same result every run."""
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    tie = (
        np.array([[2, 2, 4, 4, 0, 7, 5, 6, 0, 5]]) / 255,
        np.array([[4, 2, 6, 3, 1, 6, 8, 8, 8, 0]]) / 255,
    )
    rng = np.random.default_rng(13)
    near_constant = 0.1 + rng.random((2, 12)) * 1e-14, 0.1 + rng.random((2, 12)) * 1e-14
    for a, b in (tie, near_constant):
        masks = np.ones(a.shape, int)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            first = get_correlation_costes(a, b, masks, scale_max=255)
        assert not [w for w in recorded if "nearly constant" in str(w.message)]
        again = get_correlation_costes(a, b, masks, scale_max=255)
        for key in first:
            np.testing.assert_array_equal(first[key], again[key])


@pytest.mark.parametrize("mode", ["Fast", "Accurate"])
def test_scipy_ties_two_pair_background_keeps_scipy_warnings(mode):
    """A two-pair background that SciPy finds near-constant: with scipy_ties=True
    the numba backend must replay it, keeping SciPy's warning and decision."""
    import warnings

    from cp_measure.core.numba.measurecolocalization import (
        get_correlation_costes as nb_costes,
    )

    fi = np.array([0.02, 0.02 + 1e-14, 0.3, 0.5, 0.7, 0.9])
    si = np.array([0.05, 0.06, 0.32, 0.48, 0.71, 0.88])
    labels = np.ones((1, 6), np.int32)
    runs = []
    for fn, extra in (
        (ref.get_correlation_costes, {}),
        (nb_costes, {"scipy_ties": True}),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = fn(
                fi[None], si[None], labels, fast_costes=mode, scale_max=1000, **extra
            )
        runs.append((out, [(w.category, str(w.message)) for w in caught]))
    (expected, expected_warnings), (got, got_warnings) = runs
    assert got_warnings == expected_warnings
    for key in expected:
        np.testing.assert_array_equal(got[key], expected[key])
