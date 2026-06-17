"""Tests for the benchmark fixture/runner/comparator (cp_measure._bench)."""

import json
import time

import numpy
import pytest

from cp_measure._bench import compare, fixtures, run

SMOKE = fixtures.SMOKE_MATRIX


# --- fixtures.py -----------------------------------------------------------------------------


def test_build_fixtures_manifest_and_roundtrip(tmp_path):
    manifest = fixtures.build_fixtures(tmp_path, SMOKE)
    assert manifest["synth_version"] == fixtures.synth.__version__
    keys = {e["key"] for e in manifest["fixtures"]}
    assert keys == {"s128_n4_seed0", "s128_n8_seed0"}
    entry = manifest["fixtures"][0]
    labels, channels = fixtures.load_fixture(tmp_path, entry)  # verifies sha256
    assert labels.shape == (128, 128) and channels.shape == (2, 128, 128)


def test_fixtures_are_deterministic(tmp_path):
    a = fixtures.build_fixtures(tmp_path / "a", SMOKE)
    b = fixtures.build_fixtures(tmp_path / "b", SMOKE)
    sha = lambda m: {e["key"]: e["sha256"] for e in m["fixtures"]}  # noqa: E731
    assert sha(a) == sha(b)


def test_load_fixture_detects_corruption(tmp_path):
    manifest = fixtures.build_fixtures(tmp_path, SMOKE)
    entry = manifest["fixtures"][0]
    entry = {**entry, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fixtures.load_fixture(tmp_path, entry)


# --- run.py ----------------------------------------------------------------------------------


def test_enumerate_covers_core_and_correlation():
    funcs = run.enumerate_functions()
    # base labels (sans any [legacy] suffix) must equal the live registries.
    from cp_measure import bulk

    expected = set(bulk.get_core_measurements()) | set(
        bulk.get_correlation_measurements()
    )
    assert {f.label for f in funcs if "[legacy]" not in f.label} == expected
    arity = {f.label: f.arity for f in funcs}
    assert arity["intensity"] == 1 and arity["pearson"] == 2


def test_run_times_all_functions(tmp_path):
    fixtures.build_fixtures(tmp_path, SMOKE)
    out = run.run(tmp_path, tmp_path / "r.json", warmup=0, reps=1, timeout=60)
    assert out["meta"]["threads"] == "1"
    # every function timed on every fixture, and texture (needs [0,1] norm) is among the ok cells.
    statuses = [c["status"] for fn in out["results"].values() for c in fn.values()]
    assert statuses and all(s == "ok" for s in statuses)
    assert out["results"]["texture"]["s128_n4_seed0"]["status"] == "ok"


def test_time_call_records_error_and_timeout():
    class _F:
        def __init__(self, fn):
            self.fn, self.kwargs = fn, {}

    def boom(*a):
        raise RuntimeError("nope")

    res = run.time_call(_F(boom), (), warmup=0, reps=1, timeout=5)
    assert res["status"] == "error" and "RuntimeError" in res["error"]

    res = run.time_call(
        _F(lambda *a: time.sleep(1.0)), (), warmup=0, reps=1, timeout=0.05
    )
    assert res["status"] == "timeout"


def test_norm01_maps_to_unit_range():
    img = numpy.array([[2.0, 4.0], [6.0, 10.0]])
    n = run._norm01(img)
    assert n.min() == 0.0 and n.max() == 1.0


# --- compare.py ------------------------------------------------------------------------------


def _report(times: dict, synth_version="0.2.0"):
    """Build a minimal run-report JSON from {function: {key: seconds}} (single fixture per cell)."""
    fix = [{"key": "s128_n4_seed0", "size": 128, "n_objects": 4, "seed": 0}]
    results = {
        fn: {
            k: {"status": "ok", "min": s, "median": s, "reps": [s]}
            for k, s in cells.items()
        }
        for fn, cells in times.items()
    }
    return {
        "meta": {"synth_version": synth_version, "reps": 1, "warmup": 0},
        "fixtures": fix,
        "results": results,
    }


def test_compare_speedup_and_classification():
    base = _report(
        {
            "a": {"s128_n4_seed0": 0.010},
            "b": {"s128_n4_seed0": 0.010},
            "c": {"s128_n4_seed0": 0.010},
        }
    )
    head = _report(
        {
            "a": {"s128_n4_seed0": 0.005},
            "b": {"s128_n4_seed0": 0.020},
            "c": {"s128_n4_seed0": 0.0101},
        }
    )
    rows = {r["function"]: r for r in compare.compare(base, head)}
    assert (
        rows["a"]["speedup"] == pytest.approx(2.0) and rows["a"]["status"] == "faster"
    )
    assert (
        rows["b"]["speedup"] == pytest.approx(0.5) and rows["b"]["status"] == "slower"
    )
    assert rows["c"]["status"] == "≈"  # within the noise band


def test_compare_new_and_removed_and_nodata():
    base = _report({"gone": {"s128_n4_seed0": 0.01}, "err": {"s128_n4_seed0": 0.01}})
    head = _report({"fresh": {"s128_n4_seed0": 0.01}, "err": {"s128_n4_seed0": 0.01}})
    # make head's "err" a failed cell
    head["results"]["err"]["s128_n4_seed0"] = {"status": "error", "error": "x"}
    rows = {r["function"]: r for r in compare.compare(base, head)}
    assert rows["fresh"]["status"] == "new"
    assert rows["gone"]["status"] == "removed"
    assert rows["err"]["status"] == "no-data"


def test_render_markdown_has_legend_and_rows():
    base = _report({"a": {"s128_n4_seed0": 0.01}})
    head = _report({"a": {"s128_n4_seed0": 0.005}})
    md = compare.render_markdown(
        compare.compare(base, head), base["meta"], head["meta"]
    )
    assert "speedup = main/head" in md
    assert "`a`" in md and "2.00×" in md
    assert "| status |" in md  # the status column is labelled


def test_compare_cli(tmp_path):
    base = _report({"a": {"s128_n4_seed0": 0.01}})
    head = _report({"a": {"s128_n4_seed0": 0.005}})
    (tmp_path / "base.json").write_text(json.dumps(base))
    (tmp_path / "head.json").write_text(json.dumps(head))
    compare.main(
        [
            "--base",
            str(tmp_path / "base.json"),
            "--head",
            str(tmp_path / "head.json"),
            "--md",
            str(tmp_path / "out.md"),
        ]
    )
    assert "2.00×" in (tmp_path / "out.md").read_text()
