"""Keep benchmark callers compatible with explicit Costes grids and older APIs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cp_measure import bulk

ROOT = Path(__file__).resolve().parents[1]


def load_script(path):
    spec = importlib.util.spec_from_file_location("costes_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def benchmark():
    return load_script(ROOT / ".github/scripts/benchmark.py")


def test_pr_benchmark_costes_uses_explicit_scale(benchmark):
    fn = next(fn for name, fn, _ in benchmark._functions() if name == "costes")
    cases = json.loads(
        (ROOT / "test/data/costes_cellprofiler_reference.json").read_text()
    )["cases"]
    case = next(c for c in cases if c["scale_max"] == 65535 and c["mode"] == "Faster")
    first = np.array(case["raw_first"], float).reshape(8, 16) / 65535
    second = np.array(case["raw_second"], float).reshape(8, 16) / 65535
    got = fn(first, second, np.ones(first.shape, dtype=np.int32))
    np.testing.assert_allclose(
        np.array(list(got.values())).ravel(), case["fractions"], rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("supports_scale", [True, False])
def test_pr_benchmark_records_grid_for_new_and_old_api(
    benchmark, monkeypatch, tmp_path, supports_scale
):
    calls = []

    def current(first, second, labels, *, scale_max=None):
        calls.append(scale_max)

    def legacy(first, second, labels):
        calls.append(1)

    def call_once(fn, args):
        fn(*args)
        return {"status": "ok", "reps": [0.001]}

    monkeypatch.setattr(bulk, "get_core_measurements", lambda: {})
    monkeypatch.setattr(
        bulk,
        "get_correlation_measurements",
        lambda: {"costes": current if supports_scale else legacy},
    )
    monkeypatch.setattr(
        benchmark, "MATRIX", {"sizes": (8,), "counts": (1,), "seeds": (0,)}
    )
    monkeypatch.setattr(benchmark, "_time", call_once)
    output = tmp_path / "benchmark.json"
    benchmark.run(str(output))
    expected = 65535 if supports_scale else 1
    assert calls == [expected]
    assert json.loads(output.read_text())["costes_scale_max"] == expected


@pytest.mark.parametrize(
    "base_scale,head_scale,filter_scale,omit_costes",
    [
        (None, 65535, None, True),
        (65535, 65535, None, False),
        (65535, 65535, 1, True),
    ],
)
def test_pr_benchmark_only_compares_matching_grids(
    benchmark, base_scale, head_scale, filter_scale, omit_costes
):
    def result(scale, duration):
        record = {
            "cells": [{"key": "a", "size": 32, "n_objects": 1}],
            "results": {
                name: {"a": {"status": "ok", "reps": [duration]}}
                for name in ["costes", "pearson"]
            },
        }
        if scale is not None:
            record["costes_scale_max"] = scale
        return record

    base, head = result(base_scale, 0.1), result(head_scale, 0.2)
    filtering = result(filter_scale, 0.1) if filter_scale is not None else None
    report = benchmark.compare(base, head, filter_base=filtering)
    assert ("#### `costes`" not in report) == omit_costes
    assert ("Costes timing comparison omitted" in report) == omit_costes
    assert "#### `pearson`" in report


def test_data_benchmark_forwards_its_normalization_scale():
    pytest.importorskip("pyarrow")
    benchmark = load_script(ROOT / "benchmarks/benchmark.py")
    config = benchmark.build_config({"correlation_costes"})
    assert config["correlation_costes"]
    assert config["correlation_costes_params"] == {"scale_max": 65535}
