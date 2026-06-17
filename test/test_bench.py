"""Tests for the benchmark fixture/runner/comparator (cp_measure._bench)."""

import time

import pytest

from cp_measure._bench import compare, fixtures, run

SMOKE = fixtures.SMOKE_MATRIX


def test_fixtures_build_load_and_determinism(tmp_path):
    a = fixtures.build_fixtures(tmp_path / "a", SMOKE)
    b = fixtures.build_fixtures(tmp_path / "b", SMOKE)
    assert a["synth_version"] == fixtures.synth.__version__
    sha = lambda m: {e["key"]: e["sha256"] for e in m["fixtures"]}  # noqa: E731
    assert sha(a) == sha(b)  # deterministic across builds
    labels, channels = fixtures.load_fixture(
        tmp_path / "a", a["fixtures"][0]
    )  # verifies sha256
    assert labels.shape == (128, 128) and channels.shape == (2, 128, 128)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fixtures.load_fixture(tmp_path / "a", {**a["fixtures"][0], "sha256": "0" * 64})


def test_run_enumerates_and_times_all_functions(tmp_path):
    from cp_measure import bulk

    expected = set(bulk.get_core_measurements()) | set(
        bulk.get_correlation_measurements()
    )
    assert {
        f.label for f in run.enumerate_functions() if "[legacy]" not in f.label
    } == expected
    fixtures.build_fixtures(tmp_path, SMOKE)
    out = run.run(tmp_path, tmp_path / "r.json", warmup=0, reps=1, timeout=60)
    statuses = [c["status"] for fn in out["results"].values() for c in fn.values()]
    assert statuses and all(
        s == "ok" for s in statuses
    )  # incl. texture (needs [0,1] norm)


def test_time_call_error_and_timeout():
    class _F:
        def __init__(self, fn):
            self.fn, self.kwargs = fn, {}

    def boom(*a):
        raise RuntimeError("nope")

    assert run.time_call(_F(boom), (), 0, 1, 5)["status"] == "error"
    assert (
        run.time_call(_F(lambda *a: time.sleep(1)), (), 0, 1, 0.05)["status"]
        == "timeout"
    )


def _report(times):
    fix = [{"key": "k", "size": 128, "n_objects": 4, "seed": 0}]
    results = {
        fn: {"k": {"status": "ok", "min": s, "median": s}} for fn, s in times.items()
    }
    return {
        "meta": {"synth_version": "0", "reps": 1},
        "fixtures": fix,
        "results": results,
    }


def test_compare_classifies_and_renders():
    base = _report(
        {"fast": 0.010, "slow": 0.010, "same": 0.010, "gone": 0.010, "err": 0.010}
    )
    head = _report(
        {"fast": 0.005, "slow": 0.020, "same": 0.0101, "fresh": 0.010, "err": 0.010}
    )
    head["results"]["err"]["k"] = {"status": "error", "error": "x"}
    rows = {r["function"]: r for r in compare.compare(base, head)}
    assert rows["fast"]["status"] == "faster" and rows["fast"][
        "speedup"
    ] == pytest.approx(2.0)
    assert rows["slow"]["status"] == "slower"
    assert rows["same"]["status"] == "≈"
    assert rows["fresh"]["status"] == "new" and rows["gone"]["status"] == "removed"
    assert rows["err"]["status"] == "no-data"
    md = compare.render_markdown(
        compare.compare(base, head), base["meta"], head["meta"]
    )
    assert "speedup = main/head" in md and "| status |" in md and "2.00×" in md
