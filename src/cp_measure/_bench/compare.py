"""Compare two ``run.py`` JSON outputs into a timing table.

Per (function, size, count) it shows the ``main`` and ``head`` time as mean (min–max) over
reps×seeds and the raw ratio ``main/head``. No pass/fail classification and no normalisation —
read the function you changed and judge the spread from its own min–max. Functions present on only
one side are flagged ``new``/``removed``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _cell_groups(report: dict) -> dict:
    """Map (size, n_objects) -> [fixture keys] from a report's fixtures list."""
    groups: dict[tuple[int, int], list[str]] = {}
    for e in report["fixtures"]:
        groups.setdefault((e["size"], e["n_objects"]), []).append(e["key"])
    return groups


def _stats(results_for_fn: dict, keys: list[str]):
    """(mean, min, max) in ms over all ok rep times of a cell, or None if none timed."""
    times = [
        t
        for k in keys
        if results_for_fn.get(k, {}).get("status") == "ok"
        for t in results_for_fn[k]["reps"]
    ]
    if not times:
        return None
    return statistics.mean(times) * 1e3, min(times) * 1e3, max(times) * 1e3


def compare(base: dict, head: dict) -> list[dict]:
    groups = _cell_groups(head)  # head defines the matrix
    base_r, head_r = base["results"], head["results"]
    rows = []
    for fn in sorted(head_r):
        for (size, n_objects), keys in sorted(groups.items()):
            m = _stats(base_r.get(fn, {}), keys)
            h = _stats(head_r[fn], keys)
            rows.append(
                {
                    "function": fn,
                    "size": size,
                    "n_objects": n_objects,
                    "main": m,
                    "head": h,
                    "speedup": (m[0] / h[0]) if (m and h) else None,
                    "note": "" if fn in base_r else "new",
                }
            )
    for fn in sorted(set(base_r) - set(head_r)):
        rows.append(
            {
                "function": fn,
                "size": None,
                "n_objects": None,
                "main": None,
                "head": None,
                "speedup": None,
                "note": "removed",
            }
        )
    return rows


def _ms(stat) -> str:
    return "—" if stat is None else f"{stat[0]:.1f} ({stat[1]:.1f}–{stat[2]:.1f})"


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
        f"time = mean (min–max) ms over reps×seeds · `speedup = main/head` · "
        f"synth v{head_meta.get('synth_version')} · {head_meta.get('n_fixtures')} fixtures "
        f"({scope}) · reps={head_meta.get('reps')}, threads=1",
        "",
        "| function | size | objects | main (ms) | head (ms) | speedup |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        label = f"`{r['function']}`" + (f" _{r['note']}_" if r["note"] else "")
        sz = "—" if r["size"] is None else r["size"]
        no = "—" if r["n_objects"] is None else r["n_objects"]
        sp = "—" if r["speedup"] is None else f"{r['speedup']:.2f}×"
        lines.append(
            f"| {label} | {sz} | {no} | {_ms(r['main'])} | {_ms(r['head'])} | {sp} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Compare two run.py JSON outputs into a timing table."
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
