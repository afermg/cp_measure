"""The PR benchmark's batch pass records and reports timings; it asserts no speed."""

import importlib.util
import json
from pathlib import Path

import pytest

from cp_measure import bulk

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def benchmark(monkeypatch):
    path = ROOT / ".github/scripts/benchmark.py"
    spec = importlib.util.spec_from_file_location("batch_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "BATCH", {"images": 3, "size": 16, "count": 2})
    return module


def test_batch_pass_times_loop_and_batch_and_skips_single_only(
    benchmark, monkeypatch, tmp_path
):
    calls = []

    def batched(masks, pixels):
        calls.append("list" if isinstance(masks, list) else "single")
        return [{}] * len(masks) if isinstance(masks, list) else {}

    def single_only(masks, pixels):
        if isinstance(masks, list):
            raise TypeError("no batch")
        return {}

    monkeypatch.setattr(
        bulk, "get_core_measurements", lambda: {"a": batched, "b": single_only}
    )
    monkeypatch.setattr(bulk, "get_correlation_measurements", lambda: {})
    output = tmp_path / "batch.json"
    benchmark.run_batch(str(output))
    record = json.loads(output.read_text())

    assert record["batch"] == benchmark.BATCH
    a, b = record["results"]["a"], record["results"]["b"]
    assert a["status"] == "ok"
    assert a["loop"]["status"] == a["batch"]["status"] == "ok"
    assert b["status"] == "unsupported" and "no batch" in b["error"]
    # Probe and batch runs pass lists; the loop passes one image at a time.
    runs = benchmark.WARMUP + benchmark.REPS
    assert calls.count("list") == 2 * runs
    assert calls.count("single") == 3 * runs


def test_batch_report_lists_every_supported_function(benchmark):
    def record(loop, batch):
        return {
            "status": "ok",
            "loop": {"status": "ok", "reps": [loop]},
            "batch": {"status": "ok", "reps": [batch]},
        }

    head = {
        "batch": {"images": 8, "size": 512, "count": 64},
        "threads": 4,
        "results": {
            "fast": record(0.4, 0.1),
            "flat": record(0.2, 0.2),
            "single": {"status": "unsupported"},
        },
    }
    base = {"results": {"fast": record(0.4, 0.3)}}  # "flat" is new on head
    report = benchmark.compare_batch(base, head, "abcdef123")

    assert "`abcdef1`" in report and "4 threads" in report
    assert "| `fast` | 400 | 100 | 4.00× | 3.00× |" in report
    assert "| `flat` | 200 | 200 | 1.00× | — |" in report
    assert "_No batch input: `single`._" in report
    assert "not a CI check" in report
