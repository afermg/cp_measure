"""Preserve the benchmark grid while adding the 128-object workload."""

import importlib.util
from pathlib import Path

import numpy as np


def test_benchmark_grid_includes_128_without_removing_64():
    path = Path(__file__).resolve().parents[1] / ".github/scripts/benchmark.py"
    spec = importlib.util.spec_from_file_location("ci_benchmark_test", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    assert benchmark.MATRIX == {
        "sizes": (256, 512, 1024),
        "counts": (16, 64, 128, 256),
        "seeds": (0, 1, 2),
    }
    # The smallest image must still contain all 128 distinct objects.
    labels, channels = benchmark.generate(256, 128, seed=0)
    np.testing.assert_array_equal(np.unique(labels), np.arange(129))
    assert channels.shape == (2, 256, 256)
