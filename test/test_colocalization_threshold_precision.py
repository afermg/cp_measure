"""Explicit float64 arithmetic, distinct from historical CP float32 rounding."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cp_measure.core import measurecolocalization as coloc

REFERENCE = json.loads(
    (
        Path(__file__).parent / "data/colocalization_cellprofiler_reference.json"
    ).read_text()
)
CASE = REFERENCE["cases"][0]


def inputs(dtype=np.float32):
    return (
        np.asarray(CASE["first"], dtype) / dtype(65535),
        np.asarray(CASE["second"], dtype) / dtype(65535),
        np.asarray(CASE["labels"], np.int32),
    )


def double_reference(first, second, metric):
    # Independent scalar-double oracle on the *stored float32 values*, not
    # renormalized acquisition integers or the production pair functions.
    a, b = ([float(x) for x in image.ravel()] for image in (first, second))
    ta, tb = 0.15 * max(a), 0.15 * max(b)
    selected = [(x, y) for x, y in zip(a, b) if x >= ta and y >= tb]
    if metric == "overlap":
        aa = math.fsum(x * x for x, _ in selected)
        bb = math.fsum(y * y for _, y in selected)
        ab = math.fsum(x * y for x, y in selected)
        return [ab / math.sqrt(aa * bb), ab / aa, ab / bb]
    weights = [1.0] * len(selected)
    if metric == "rwc":
        ra, rb = sorted(set(a)), sorted(set(b))
        ranks = max(len(ra), len(rb))
        weights = [
            (ranks - abs(ra.index(x) - rb.index(y))) / ranks for x, y in selected
        ]
    return [
        math.fsum(x * w for (x, _), w in zip(selected, weights))
        / math.fsum(x for x in a if x >= ta),
        math.fsum(y * w for (_, y), w in zip(selected, weights))
        / math.fsum(y for y in b if y >= tb),
    ]


@pytest.mark.parametrize("metric", ["manders_fold", "overlap", "rwc"])
@pytest.mark.parametrize(
    "backend,single",
    [("numpy", False), ("numpy", True), ("numba", False)],
    ids=["multi-label", "single-mask", "numba-multi-label"],
)
@pytest.mark.parametrize("precision", [None, "input", "float64"])
def test_float32_boundary_precision_modes(metric, backend, single, precision):
    first, second, labels = inputs()
    expected = double_reference(first, second, metric)
    first.setflags(write=False)
    second.setflags(write=False)
    module = coloc
    if backend == "numba":
        pytest.importorskip("numba")
        from cp_measure.core.numba import measurecolocalization as module
    function = getattr(module, "get_correlation_" + metric + ("_ind" if single else ""))
    result = function(
        first,
        second,
        labels.astype(bool) if single else labels,
        thr=np.float32(15),
        **({} if precision is None else {"precision": precision}),
    )
    for (key, values), value in zip(result.items(), expected, strict=True):
        expected_key = (
            key[:-2] + ("_A_B" if key.endswith("_1") else "_B_A")
            if key.endswith(("_1", "_2"))
            else key + "_A_B"
        )
        historical = CASE["expected"][expected_key]
        # Pre-#129 Manders sums/division round in float32, whereas CP's stored
        # final coefficient is double. Preserve that distinction explicitly.
        input_expected = (
            np.asarray(historical, np.float32)
            if metric == "manders_fold"
            else historical
        )
        np.testing.assert_allclose(
            np.asarray(values).reshape(-1),
            [value] if precision == "float64" else input_expected,
            rtol=1e-12,
            atol=1e-12,
        )
        # The two policies must actually distinguish this boundary fixture.
        assert not np.allclose([value], historical, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("scale", [1e20, 1e-30])
def test_float32_products_do_not_overflow_or_underflow(backend, scale):
    module = coloc
    if backend == "numba":
        pytest.importorskip("numba")
        from cp_measure.core.numba import measurecolocalization as module
    pixels = np.array([[scale, 2 * scale]], dtype=np.float32)
    result = module.get_correlation_overlap(
        pixels, pixels, np.ones(pixels.shape, np.int32), precision="float64"
    )
    np.testing.assert_allclose(
        list(result.values()), [[1.0]] * 3, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.longdouble, np.uint16]
)
@pytest.mark.parametrize("precision", ["input", "float64"])
def test_precision_promotion_is_exact_and_preserves_copy_policy(dtype, precision):
    from cp_measure._precision import promote_float_pixels

    pixels = np.array([1, 2, 3], dtype=dtype)
    pixels.setflags(write=False)
    result = promote_float_pixels(pixels, precision)
    np.testing.assert_array_equal(result, pixels)
    assert not pixels.flags.writeable
    if (
        precision == "float64"
        and pixels.dtype.kind == "f"
        and pixels.dtype.itemsize < 8
    ):
        assert result.dtype == np.float64
        assert not np.shares_memory(result, pixels)
    else:
        assert result is pixels


@pytest.mark.parametrize("dtype", [np.float64, np.uint16])
def test_float64_and_integer_threshold_membership_unchanged(dtype):
    first = np.array([3, 20, 10, 0], dtype=dtype)
    second = first[::-1].copy()
    tf, ts = 0.15 * first.max(), 0.15 * second.max()
    joint = (first >= tf) & (second >= ts)
    expected = (
        first[joint].sum() / first[first >= tf].sum(),
        second[joint].sum() / second[second >= ts].sum(),
    )
    np.testing.assert_array_equal(coloc._manders_pair(first, second, 15), expected)
