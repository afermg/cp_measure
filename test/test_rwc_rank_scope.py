"""RWC's object-local default and explicit CellProfiler object-set ranking."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_measure._sanitize import sanitize
from cp_measure.core.measurecolocalization import get_correlation_rwc, _rwc_pair

REFERENCE = json.loads(
    (
        Path(__file__).parent / "data/colocalization_cellprofiler_reference.json"
    ).read_text()
)


def scene(case_index=1):
    case = REFERENCE["cases"][case_index]
    return (
        np.asarray(case["first"], np.float32) / np.float32(65535),
        np.asarray(case["second"], np.float32) / np.float32(65535),
        np.asarray(case["labels"], np.int32),
    )


@pytest.mark.parametrize(
    "case_index", [0, 1], ids=["threshold-boundary", "multi-object-tied-ranks"]
)
def test_object_set_scope_matches_cellprofiler(case_index):
    first, second, labels = scene(case_index)
    if case_index == 1:
        labels = labels.astype(np.uint64)
    result = get_correlation_rwc(first, second, labels, rank_scope="object_set")
    expected = REFERENCE["cases"][case_index]["expected"]
    for suffix, channels in (("1", "A_B"), ("2", "B_A")):
        np.testing.assert_allclose(
            result["Correlation_RWC_" + suffix],
            expected["Correlation_RWC_" + channels],
            rtol=1e-7,
            atol=1e-10,
        )


def test_default_is_unchanged_object_local_ranking():
    first, second, labels = scene()
    default = get_correlation_rwc(first, second, labels)
    explicit = get_correlation_rwc(first, second, labels, rank_scope="object")
    pairs = [_rwc_pair(first[labels == k], second[labels == k], 15) for k in (1, 2)]
    np.testing.assert_array_equal(default["Correlation_RWC_1"], [p[0] for p in pairs])
    np.testing.assert_array_equal(default["Correlation_RWC_2"], [p[1] for p in pairs])
    assert default == explicit
    assert default != get_correlation_rwc(
        first, second, labels, rank_scope="object_set"
    )


def test_object_set_ignores_background_but_depends_on_other_objects():
    first, second, labels = scene()
    local = get_correlation_rwc(first, second, labels)
    global_before = get_correlation_rwc(first, second, labels, rank_scope="object_set")
    first[labels == 0] = np.nan
    second[labels == 0] = np.inf
    assert global_before == get_correlation_rwc(
        first, second, labels, rank_scope="object_set"
    )
    first[labels == 2] = 0.85
    second[labels == 2] = 0.85
    local_after = get_correlation_rwc(first, second, labels)
    global_after = get_correlation_rwc(first, second, labels, rank_scope="object_set")
    for key in local:
        assert local[key][0] == local_after[key][0]
        assert global_before[key][0] != global_after[key][0]


def test_scope_preserves_empty_absent_and_zero_signal_conventions():
    first, second, labels = scene()
    assert get_correlation_rwc(
        first, second, np.zeros_like(labels), rank_scope="object_set"
    ) == {"Correlation_RWC_1": [], "Correlation_RWC_2": []}
    labels[labels == 2] = 3
    result = get_correlation_rwc(first, second, labels, rank_scope="object_set")
    for values in result.values():
        assert len(values) == 3
        assert values[1] == 0
    result = get_correlation_rwc(
        np.zeros_like(first), np.zeros_like(second), labels, rank_scope="object_set"
    )
    for values in result.values():
        assert np.isnan(values[0]) and np.isnan(values[2])
        assert values[1] == 0


def test_no_joint_threshold_pixels_returns_zero():
    first = np.array([[0.0, 1.0]])
    second = 1 - first
    result = get_correlation_rwc(
        first, second, np.ones_like(first, dtype=int), rank_scope="object_set"
    )
    assert result == {"Correlation_RWC_1": [0.0], "Correlation_RWC_2": [0.0]}


def test_sanitized_noncontiguous_ids_preserve_scope():
    first, second, labels = scene()
    dense = get_correlation_rwc(first, second, labels, rank_scope="object_set")
    labels[labels == 1] = 42
    labels[labels == 2] = 99
    assert (
        sanitize(get_correlation_rwc)(first, second, labels, rank_scope="object_set")
        == dense
    )


def test_bulk_forwards_explicit_scope():
    from cp_measure.bulk import get_correlation_measurements

    first, second, labels = scene()
    result = get_correlation_measurements()["rwc"](
        first, second, labels, rank_scope="object_set"
    )
    assert result == get_correlation_rwc(first, second, labels, rank_scope="object_set")


def test_featurizer_forwards_explicit_scope():
    from cp_measure.featurizer import featurize, make_featurizer_config
    from conftest import ALL_OFF

    first, second, labels = scene()
    config = make_featurizer_config(
        ["A", "B"],
        **{**ALL_OFF, "correlation_rwc": True},
        correlation_rwc_params={"rank_scope": "object_set"},
    )
    data, columns, rows = featurize(
        np.stack([first, second])[None], labels[None, None], config
    )
    expected = get_correlation_rwc(first, second, labels, rank_scope="object_set")
    assert len(rows) == 2 and len(columns) == 2
    for index, column in enumerate(columns):
        key = next(key for key in expected if key in column)
        np.testing.assert_allclose(data[:, index], expected[key], rtol=1e-7, atol=1e-10)


def test_invalid_scope_and_nonfinite_foreground_are_rejected():
    first, second, labels = scene()
    with pytest.raises(ValueError, match="rank_scope"):
        get_correlation_rwc(first, second, labels, rank_scope="image")
    first[labels == 1] = np.nan
    with pytest.raises(ValueError, match="finite foreground"):
        get_correlation_rwc(first, second, labels, rank_scope="object_set")
