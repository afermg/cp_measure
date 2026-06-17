"""Compare two ``run.py`` JSON outputs (main vs PR head) into a speedup table.

``python -m cp_measure._bench.compare --base main.json --head head.json [--md out.md]``

Speedup is ``base_time / head_time`` (>1 = head is faster, <1 = head is a regression). Per
(function, size, count) cell the per-fixture **min** time is taken (noise floor), then the
**median** across seeds. A function untouched by the PR lands within the noise band (≈) — that is
the "what changed" signal, no change-detection needed. Functions present only on head are "new";
only on base are "removed".
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

NOISE_BAND = 0.05  # |speedup - 1| within this is reported "≈" (within timing noise)


def _cell_groups(report: dict) -> dict:
    """Map (size, n_objects) -> [fixture keys], from a report's fixtures list."""
    groups: dict[tuple[int, int], list[str]] = {}
    for e in report["fixtures"]:
        groups.setdefault((e["size"], e["n_objects"]), []).append(e["key"])
    return groups


def _aggregate(results_for_fn: dict, keys: list[str]) -> float | None:
    """Median across seeds of each fixture's min time; None if no fixture timed ok."""
    mins = [
        results_for_fn[k]["min"]
        for k in keys
        if results_for_fn.get(k, {}).get("status") == "ok"
    ]
    return statistics.median(mins) if mins else None


def compare(base: dict, head: dict) -> list[dict]:
    groups = _cell_groups(head)  # head defines the matrix
    base_results, head_results = base["results"], head["results"]
    rows = []
    for fn in sorted(head_results):
        for (size, n_objects), keys in sorted(groups.items()):
            head_t = _aggregate(head_results[fn], keys)
            base_t = _aggregate(base_results.get(fn, {}), keys)
            if fn not in base_results:
                status = "new"
                speedup = None
            elif head_t is None or base_t is None:
                status = "no-data"  # error/timeout on at least one side
                speedup = None
            else:
                speedup = base_t / head_t
                status = (
                    "≈"
                    if abs(speedup - 1) <= NOISE_BAND
                    else ("faster" if speedup > 1 else "slower")
                )
            rows.append(
                {
                    "function": fn,
                    "size": size,
                    "n_objects": n_objects,
                    "base_ms": None if base_t is None else base_t * 1e3,
                    "head_ms": None if head_t is None else head_t * 1e3,
                    "speedup": speedup,
                    "status": status,
                }
            )
    # Functions removed on head (present in base only).
    for fn in sorted(set(base_results) - set(head_results)):
        rows.append(
            {
                "function": fn,
                "size": None,
                "n_objects": None,
                "base_ms": None,
                "head_ms": None,
                "speedup": None,
                "status": "removed",
            }
        )
    return rows


def render_markdown(rows: list[dict], base_meta: dict, head_meta: dict) -> str:
    def fmt(x, spec):
        return "—" if x is None else format(x, spec)

    lines = [
        "### Benchmark — PR head vs `main`",
        "",
        f"`speedup = main_time / head_time` · **>1 = faster**, <1 = regression, "
        f"≈ = within ±{int(NOISE_BAND * 100)}% noise · synth v{head_meta.get('synth_version')} · "
        f"reps={head_meta.get('reps')}, warmup={head_meta.get('warmup')}, threads=1",
        "",
        "| function | size | objects | main (ms) | head (ms) | speedup | |",
        "|---|--:|--:|--:|--:|--:|:--|",
    ]
    emoji = {
        "faster": "🟢",
        "slower": "🔴",
        "≈": "⚪",
        "new": "🆕",
        "removed": "🗑️",
        "no-data": "⚠️",
    }
    for r in rows:
        sp = "—" if r["speedup"] is None else f"{r['speedup']:.2f}×"
        lines.append(
            f"| `{r['function']}` | {fmt(r['size'], 'd')} | {fmt(r['n_objects'], 'd')} | "
            f"{fmt(r['base_ms'], '.1f')} | {fmt(r['head_ms'], '.1f')} | {sp} | "
            f"{emoji.get(r['status'], '')} {r['status']} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Compare two run.py JSON outputs into a speedup table."
    )
    p.add_argument("--base", required=True, help="main run JSON")
    p.add_argument("--head", required=True, help="head run JSON")
    p.add_argument("--md", help="write the markdown table here (default: stdout)")
    a = p.parse_args(argv)
    base = json.loads(Path(a.base).read_text())
    head = json.loads(Path(a.head).read_text())
    md = render_markdown(compare(base, head), base["meta"], head["meta"])
    if a.md:
        Path(a.md).write_text(md)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
