"""CI's small benchmark must measure pair dispatch, not just direct functions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import cp_measure
from cp_measure._detect import HAS_NUMBA

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/benchmark_colocalization.py"
)
spec = importlib.util.spec_from_file_location("ci_colocalization", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize(
    "backend",
    [
        "numpy",
        pytest.param(
            "numba", marks=pytest.mark.skipif(not HAS_NUMBA, reason="needs numba")
        ),
    ],
)
def test_real_featurizer_benchmark(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(cp_measure, "_ACCELERATOR", cp_measure._ACCELERATOR)
    import cp_measure.featurizer as featurizer

    original = featurizer.featurize
    modes = []

    def counted(*args, **kwargs):
        modes.append(args[2]["costes_pair_mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(featurizer, "featurize", counted)
    path = tmp_path / "head.json"
    bench.run(path, backend, revision="head", reps=1)
    assert modes.count("ordered") == 2  # Once per policy, no timed repeats.
    assert modes.count("unordered") == 4  # Warmup plus one timed call per policy.
    report = bench.compare(path, path)
    assert "Correctness **PASS**" in report
    assert report.count("\n|") == 4  # Header, separator, two default-mode rows.
    assert f"|{backend}|defaults|" in report
    assert f"|{backend}|matched|" in report
    metadata = json.loads(path.read_text())
    with np.load(path.with_suffix(".npz")) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    # Model an older main without the option: default is its ordered arm.
    for policy in ("defaults", "matched"):
        assert metadata["records"][f"{policy}-ordered"]["reps_seconds"] == []
        assert (
            metadata["records"][f"{policy}-default"]["pair_executions"]["costes"] == 10
        )
        metadata["records"][f"{policy}-default"] = {
            **metadata["records"][f"{policy}-ordered"],
            "reps_seconds": metadata["records"][f"{policy}-default"]["reps_seconds"],
        }
        arrays[f"{policy}-default"] = arrays[f"{policy}-ordered"].copy()
    base = tmp_path / "main.json"
    base.write_text(json.dumps(metadata))
    np.savez_compressed(base.with_suffix(".npz"), **arrays)
    assert "Correctness **PASS**" in bench.compare(base, path)
    # A retained-value regression must fail, rather than disappear into timings.
    arrays["defaults-ordered"][0, 0] += 1
    np.savez_compressed(base.with_suffix(".npz"), **arrays)
    with pytest.raises(AssertionError):
        bench.compare(base, path)


def test_compact_combined_table(monkeypatch):
    monkeypatch.setattr(
        bench,
        "comparison_rows",
        lambda base, head: [
            f"|{head}|{policy}|1|1|1.00×|" for policy in ("defaults", "matched")
        ],
    )
    report = bench.compare(["base-numpy", "base-numba"], ["numpy", "numba"])
    assert report.count("### Featurizer colocalization") == 1
    assert report.count("\n|") == 6  # One header/separator and four rows.
    for backend in ("numpy", "numba"):
        for policy in ("defaults", "matched"):
            assert f"|{backend}|{policy}|" in report
    with pytest.raises(ValueError, match="matching nonempty"):
        bench.compare(["base"], ["head", "extra"])


def test_legacy_configuration_without_new_keyword(monkeypatch):
    import cp_measure.featurizer as featurizer

    original = featurizer.make_featurizer_config

    def legacy(channels, **kwargs):
        assert "costes_pair_mode" not in kwargs
        config = original(channels, **kwargs)
        config.pop("costes_pair_mode")
        return config

    monkeypatch.setattr(featurizer, "make_featurizer_config", legacy)
    for mode in ("default", "ordered"):
        config, effective = bench.configuration("matched", mode)
        assert "costes_pair_mode" not in config
        assert effective == "ordered"


@pytest.mark.parametrize(
    "before,after", [(np.nan, 0.0), (np.inf, -np.inf), (1.0, 1.000000000001)]
)
def test_exact_gate_rejects_changed_values_or_nonfinite_patterns(before, after):
    with pytest.raises(AssertionError):
        bench.assert_exact(np.array([[after]]), np.array([[before]]))


def test_no_missing_non_costes_columns_or_rows_allowed():
    previous = {
        "rows": [["ci", "cells", 1]],
        "columns": ["Pearson", "Slope"],
        "effective_mode": "ordered",
    }
    current = {**previous, "columns": ["Pearson"]}
    with pytest.raises(AssertionError, match="column coverage"):
        bench.assert_retained(previous, current, np.ones((1, 2)), np.ones((1, 1)))
    with pytest.raises(AssertionError, match="Row metadata"):
        bench.assert_retained(
            previous, {**previous, "rows": []}, np.ones((1, 2)), np.ones((1, 2))
        )
