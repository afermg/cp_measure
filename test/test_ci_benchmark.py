"""Representative CI workload selection and sparse comparison tables."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import cp_measure


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[1] / ".github/scripts/benchmark.py"
    spec = importlib.util.spec_from_file_location("ci_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_representative_workload_order(benchmark):
    assert benchmark.WORKLOADS == (
        ("Tiles", 256, 16),
        ("Intermediate", 512, 64),
        ("Cell Painting-like", 1024, 128),
        ("Spatial-omics-like", 1024, 256),
    )
    assert benchmark.SEEDS == (0, 1, 2)


def test_cell_painting_generator_is_seeded_and_has_128_objects(benchmark):
    labels, channels = benchmark.generate(1024, 128, seed=1)
    assert labels.shape == (1024, 1024)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(np.unique(labels), np.arange(129))
    assert channels.shape == (2, 1024, 1024)
    assert channels.dtype == np.float32
    assert np.isfinite(channels).all()
    again_labels, again_channels = benchmark.generate(1024, 128, seed=1)
    np.testing.assert_array_equal(labels, again_labels)
    np.testing.assert_array_equal(channels, again_channels)
    _, other_channels = benchmark.generate(1024, 128, seed=2)
    assert not np.array_equal(channels, other_channels)


@pytest.mark.parametrize("accelerator", [None, "numba"])
def test_run_visits_only_paired_workloads(
    benchmark, monkeypatch, tmp_path, accelerator
):
    generated, selected, arities = [], [], []

    def generate(size, count, seed):
        generated.append((size, count, seed))
        return np.ones((2, 2), np.int32), np.arange(8).reshape(2, 2, 2)

    def core(labels, pixels):
        arities.append(1)

    def costes(first, second, labels):
        arities.append(2)

    def call_once(fn, args):
        fn(*args)
        return {"status": "ok", "reps": [0.001]}

    monkeypatch.setattr(benchmark, "generate", generate)
    monkeypatch.setattr(
        benchmark, "_functions", lambda: [("intensity", core, 1), ("costes", costes, 2)]
    )
    monkeypatch.setattr(benchmark, "_time", call_once)
    monkeypatch.setattr(cp_measure, "set_accelerator", selected.append)
    path = tmp_path / "timings.json"
    benchmark.run(str(path), accelerator)
    result = json.loads(path.read_text())
    assert selected == (["numba"] if accelerator else [])
    assert generated == [
        (size, n, seed)
        for _, size, n in benchmark.WORKLOADS
        for seed in benchmark.SEEDS
    ]
    assert arities == [1, 2] * 12
    expected_cells = [
        {
            "key": f"s{size}_n{n}_seed{seed}",
            "size": size,
            "n_objects": n,
            "workload": name,
            "seed": seed,
        }
        for name, size, n in benchmark.WORKLOADS
        for seed in benchmark.SEEDS
    ]
    assert result["cells"] == expected_cells
    for values in result["results"].values():
        assert list(values) == [c["key"] for c in expected_cells]
        assert all(v["status"] == "ok" for v in values.values())


def fake_run(benchmark, durations):
    cells = [
        {
            "key": f"s{size}_n{n}_seed{seed}",
            "size": size,
            "n_objects": n,
            "workload": name,
        }
        for name, size, n in benchmark.WORKLOADS
        for seed in benchmark.SEEDS
    ]
    return {
        "cells": cells,
        "results": {
            name: {c["key"]: {"status": "ok", "reps": [duration]} for c in cells}
            for name, duration in durations.items()
        },
    }


def test_compare_reports_paired_rows_and_preserves_numba_filter(benchmark):
    base = fake_run(benchmark, {"zernike": 0.8, "texture": 0.8})
    head = fake_run(benchmark, {"zernike": 0.1, "texture": 0.1})
    filtering = fake_run(benchmark, {"zernike": 0.2, "texture": 0.1})
    report = benchmark.compare(base, head, filter_base=filtering)
    assert "#### `zernike`" in report
    assert "#### `texture`" not in report
    table = report.split("#### `zernike`")[1]
    expected = [
        f"| {name} | {size} × {size} | {n} | 8.00× |"
        for name, size, n in benchmark.WORKLOADS
    ]
    assert [line for line in table.splitlines() if "8.00×" in line] == expected
    assert "| 256 × 256 | 128 |" not in report
    assert "| 512 × 512 | 128 |" not in report


def test_unchanged_report_still_lists_workloads(benchmark):
    run = fake_run(benchmark, {"zernike": 0.1})
    report = benchmark.compare(run, run)
    assert "No function moved" in report
    assert "####" not in report
    for name, size, n in benchmark.WORKLOADS:
        assert f"| {name} | {size} × {size} | {n} |" in report


def test_compare_legacy_metadata_and_missing_cell(benchmark):
    base = fake_run(benchmark, {"zernike": 0.2})
    head = fake_run(benchmark, {"zernike": 0.1})
    for cell in head["cells"]:
        del cell["workload"]
        if cell["n_objects"] == 128:
            del base["results"]["zernike"][cell["key"]]
    report = benchmark.compare(base, head)
    assert "| Synthetic | 1024 × 1024 | 128 | — |" in report
    assert "| Synthetic | 1024 × 1024 | 256 | 2.00× |" in report
