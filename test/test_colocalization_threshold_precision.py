"""Float32 threshold membership must match the historical CP array calculation."""

import json
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


@pytest.mark.parametrize("metric", ["manders_fold", "overlap", "rwc"])
@pytest.mark.parametrize(
    "backend,single",
    [("numpy", False), ("numpy", True), ("numba", False)],
    ids=["multi-label", "single-mask", "numba-multi-label"],
)
def test_float32_threshold_boundary_matches_cellprofiler(metric, backend, single):
    first, second, labels = inputs()
    module = coloc
    if backend == "numba":
        pytest.importorskip("numba")
        from cp_measure.core.numba import measurecolocalization as module
    function = getattr(module, "get_correlation_" + metric + ("_ind" if single else ""))
    result = function(first, second, labels.astype(bool) if single else labels, thr=15)
    for key, values in result.items():
        if key.endswith("_1"):
            expected_key = key[:-2] + "_A_B"
        elif key.endswith("_2"):
            expected_key = key[:-2] + "_B_A"
        else:
            expected_key = key + "_A_B"
        np.testing.assert_allclose(
            np.asarray(values).reshape(-1),
            CASE["expected"][expected_key],
            rtol=1e-7,
            atol=1e-10,
        )


@pytest.mark.parametrize("dtype", [np.float64, np.uint16])
def test_float64_and_integer_threshold_membership_unchanged(dtype):
    # Compare to the original scalar formula; no quantization or epsilon allowed.
    first = np.array([3, 20, 10, 0], dtype=dtype)
    second = first[::-1].copy()
    tf, ts = 0.15 * first.max(), 0.15 * second.max()
    joint = (first >= tf) & (second >= ts)
    expected = (
        first[joint].sum() / first[first >= tf].sum(),
        second[joint].sum() / second[second >= ts].sum(),
    )
    np.testing.assert_array_equal(coloc._manders_pair(first, second, 15), expected)
