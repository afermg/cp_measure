"""Compare two ``run.py`` JSON outputs into a speedup table.

``speedup = base/head`` (>1 = head faster). Per (function, size, count) cell: per-fixture min, then
median across seeds. Untouched functions land in the noise band (≈) — the "what changed" signal.
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
            elif not head_t or base_t is None:  # missing/errored, or a head time of 0
                status = "no-data"
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


_EMOJI = {
    "faster": "🟢",
    "slower": "🔴",
    "≈": "⚪",
    "new": "🆕",
    "removed": "🗑️",
    "no-data": "⚠️",
}


def _fmt(x, spec="", suffix=""):
    return "—" if x is None else format(x, spec) + suffix


def render_markdown(rows: list[dict], base_meta: dict, head_meta: dict) -> str:
    m = head_meta.get("matrix") or {}
    scope = (
        f"{m.get('sizes')}×{m.get('counts')}×{len(m.get('seeds', []))} seeds"
        if m
        else "?"
    )
    lines = [
        "### Benchmark — PR head vs `main`",
        "",
        f"`speedup = main/head` · **>1 = faster**, ≈ within ±{int(NOISE_BAND * 100)}% noise · "
        f"synth v{head_meta.get('synth_version')} · {head_meta.get('n_fixtures')} fixtures "
        f"({scope}) · reps={head_meta.get('reps')}, threads=1",
        "",
        "| function | size | objects | main (ms) | head (ms) | speedup | status |",
        "|---|--:|--:|--:|--:|--:|:--|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['function']}` | {_fmt(r['size'], 'd')} | {_fmt(r['n_objects'], 'd')} | "
            f"{_fmt(r['base_ms'], '.1f')} | {_fmt(r['head_ms'], '.1f')} | {_fmt(r['speedup'], '.2f', '×')} | "
            f"{_EMOJI.get(r['status'], '')} {r['status']} |"
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
