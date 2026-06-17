#!/usr/bin/env python3
"""Self-contained PR benchmark — lives entirely in .github/scripts, nothing in the package.

Subcommands:
  run --out FILE                 generate seeded synthetic inputs, time every cp_measure get_*  -> JSON
  compare --base F --head F [--md F]   diff two run JSONs into a head-vs-main timing table

Run once per environment (PR head, main) on the SAME seeded inputs (pure-numpy generation is
deterministic), then compare. The driver installs each env; this script only needs cp_measure
importable plus numpy.
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import statistics  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter  # noqa: E402

import numpy  # noqa: E402

MATRIX = {"sizes": (256, 512, 1024, 2048), "counts": (16, 64, 256), "seeds": (0, 1, 2)}
BLOBS_PER_CHANNEL = 5
WARMUP, REPS, TIMEOUT = 1, 3, 120.0
AFFECTED = 1.1  # a function is "affected" if its best speedup reaches this


# --- synthetic generator: n ellipses on a grid + random Gaussian blobs per channel --------------
def generate(size: int, n: int, n_channels: int = 2, seed: int = 0):
    rng = numpy.random.default_rng(seed)
    yy, xx = numpy.mgrid[0:size, 0:size]
    labels = numpy.zeros((size, size), numpy.int32)
    if n:
        cols = int(numpy.ceil(numpy.sqrt(n)))
        rows = int(numpy.ceil(n / cols))
        a, b = 0.35 * size / rows, 0.35 * size / cols
        for k in range(n):
            r, c = divmod(k, cols)
            cy, cx = (r + 0.5) * size / rows, (c + 0.5) * size / cols
            labels[((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2 <= 1] = k + 1
    channels = []
    for _ in range(n_channels):
        img = numpy.zeros((size, size))
        for _ in range(BLOBS_PER_CHANNEL):
            cy, cx = rng.uniform(0, size, 2)
            s = rng.uniform(size / 10, size / 5)
            img += numpy.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
        channels.append(img.astype(numpy.float32))
    return labels, numpy.stack(channels)


# --- timing -------------------------------------------------------------------------------------
class _Timeout(Exception):
    pass


def _raise_timeout(*_):
    raise _Timeout()


@contextmanager
def _time_limit(seconds: float):
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _norm01(img):
    img = img.astype("float64")
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else img - lo


def _functions():
    from cp_measure import bulk

    out = []
    for arity, reg in (
        (1, bulk.get_core_measurements()),
        (2, bulk.get_correlation_measurements()),
    ):
        for name, fn in reg.items():
            out.append((name, fn, arity))
    return out


def _time(fn, args) -> dict:
    try:
        with _time_limit(TIMEOUT):
            for _ in range(WARMUP):
                fn(*args)
            reps = []
            for _ in range(REPS):
                t = perf_counter()
                fn(*args)
                reps.append(perf_counter() - t)
    except _Timeout:
        return {"status": "timeout"}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}
    return {"status": "ok", "reps": reps}


def run(out_path: str):
    funcs = _functions()
    cells, results = [], {name: {} for name, _, _ in funcs}
    for size in MATRIX["sizes"]:
        for n in MATRIX["counts"]:
            for seed in MATRIX["seeds"]:
                labels, channels = generate(size, n, 2, seed)
                imgs = (_norm01(channels[0]), _norm01(channels[1]))
                key = f"s{size}_n{n}_seed{seed}"
                cells.append({"key": key, "size": size, "n_objects": n})
                for name, fn, arity in funcs:
                    args = (
                        (labels, imgs[0]) if arity == 1 else (imgs[0], imgs[1], labels)
                    )
                    results[name][key] = _time(fn, args)
    Path(out_path).write_text(
        json.dumps({"cells": cells, "results": results}, indent=2)
    )


# --- compare ------------------------------------------------------------------------------------
def _median_ms(results_for_fn: dict, keys: list[str]):
    """Median (ms) over all ok rep times in a cell's seeds, or None."""
    times = [
        t
        for k in keys
        if results_for_fn.get(k, {}).get("status") == "ok"
        for t in results_for_fn[k]["reps"]
    ]
    return statistics.median(times) * 1e3 if times else None


def compare(base: dict, head: dict, commit: str = "") -> str:
    groups: dict[tuple, list[str]] = {}
    for e in head["cells"]:
        groups.setdefault((e["size"], e["n_objects"]), []).append(e["key"])
    sizes = sorted({s for s, _ in groups})
    counts = sorted({n for _, n in groups})
    br, hr = base["results"], head["results"]

    ref = f"`{commit[:7]}`" if commit else "PR head"
    out = [
        f"### Benchmark — {ref} vs `main`",
        "",
        f"`speedup = main/head` · median per cell · showing functions ≥ {AFFECTED:.1f}× faster",
    ]

    affected = []  # (function, {(size, count): speedup})
    for fn in sorted(hr):
        grid, best = {}, 0.0
        for size in sizes:
            for n in counts:
                m = _median_ms(br.get(fn, {}), groups.get((size, n), []))
                h = _median_ms(hr[fn], groups.get((size, n), []))
                grid[(size, n)] = (m / h) if (m and h) else None
                if grid[(size, n)]:
                    best = max(best, grid[(size, n)])
        if best >= AFFECTED:
            affected.append((fn, grid))

    if not affected:
        out += ["", f"_No function changed by ≥{AFFECTED:.1f}×._"]
        return "\n".join(out)

    for fn, grid in affected:
        out += [
            "",
            f"#### `{fn}`",
            "",
            "| size \\ objects | " + " | ".join(str(n) for n in counts) + " |",
            "|---" + "|--:" * len(counts) + "|",
        ]
        for size in sizes:
            row = [
                (f"{grid[(size, n)]:.2f}×" if grid.get((size, n)) else "—")
                for n in counts
            ]
            out.append(f"| **{size}** | " + " | ".join(row) + " |")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("--head", required=True)
    c.add_argument("--commit", default="")
    c.add_argument("--md")
    a = p.parse_args(argv)
    if a.cmd == "run":
        run(a.out)
    else:
        md = compare(
            json.loads(Path(a.base).read_text()),
            json.loads(Path(a.head).read_text()),
            a.commit,
        )
        (Path(a.md).write_text(md) if a.md else print(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
