#!/usr/bin/env python3
"""Self-contained PR benchmark — lives entirely in .github/scripts, nothing in the package.

Subcommands:
  run --out FILE                 generate seeded synthetic inputs, time every cp_measure get_*  -> JSON
  compare --base F --head F [--md F]   diff two run JSONs into a head-vs-main timing table
  run-batch --out FILE           time each batch-capable get_* on a batch vs a loop of single calls
  compare-batch --base F --head F [--md F]   report batch timings (head, with main's for reference)

Run once per environment (PR head, main) on the SAME seeded inputs (pure-numpy generation is
deterministic), then compare. The driver installs each env; this script only needs cp_measure
importable plus numpy.
"""

import os

# Pinned to one thread so single-image timings reflect algorithmic cost. The batch pass
# is the exception: the driver sets NUMBA_NUM_THREADS to the runner's CPUs, which sizes
# the map_images thread pool (setdefault leaves an explicit value alone).
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
import inspect  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import statistics  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter  # noqa: E402

import numpy  # noqa: E402

MATRIX = {"sizes": (256, 512, 1024), "counts": (16, 64, 128, 256), "seeds": (0, 1, 2)}
BLOBS_PER_CHANNEL = 5
WARMUP, REPS, TIMEOUT = 1, 3, 120.0
# Batch pass: one batch of distinct same-size images, as a plate site would give.
BATCH = {"images": 8, "size": 512, "count": 64}
AFFECTED = 1.05  # report a function if any cell moves by this factor either way (faster or slower)


# --- synthetic generator: n ellipses on a grid + random Gaussian blobs per channel --------------
def generate(size: int, n: int, seed: int = 0):
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
    for _ in range(2):  # ch0 → core features, ch0+ch1 → colocalisation
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
            # This driver also runs against main/older releases. Synthetic
            # normalized floats use an explicit 16-bit grid where supported.
            if name == "costes" and "scale_max" in inspect.signature(fn).parameters:
                fn = partial(fn, scale_max=65535)
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


def run(out_path: str, accelerator: str | None = None):
    if accelerator:
        import cp_measure

        cp_measure.set_accelerator(accelerator)
    funcs = _functions()
    cells, results = [], {name: {} for name, _, _ in funcs}
    for size in MATRIX["sizes"]:
        for n in MATRIX["counts"]:
            for seed in MATRIX["seeds"]:
                labels, channels = generate(size, n, seed)
                imgs = (_norm01(channels[0]), _norm01(channels[1]))
                key = f"s{size}_n{n}_seed{seed}"
                cells.append({"key": key, "size": size, "n_objects": n})
                for name, fn, arity in funcs:
                    args = (
                        (labels, imgs[0]) if arity == 1 else (imgs[0], imgs[1], labels)
                    )
                    results[name][key] = _time(fn, args)
    costes = next(fn for name, fn, _ in funcs if name == "costes")
    scale = getattr(costes, "keywords", {}).get("scale_max", 1)
    Path(out_path).write_text(
        json.dumps(
            {"cells": cells, "results": results, "costes_scale_max": scale}, indent=2
        )
    )


def _threads() -> int:
    try:
        from cp_measure.primitives.parallel import _usable_cpus
    except ImportError:  # older releases have no batch thread pool
        return 1
    return _usable_cpus()


def run_batch(out_path: str, accelerator: str | None = None):
    """Time each batch-capable function on one batch vs a loop of single-image calls.

    Batch support is probed on a two-image batch; functions that reject a list are
    recorded as unsupported and skipped. Results are reported, never asserted.
    """
    if accelerator:
        import cp_measure

        cp_measure.set_accelerator(accelerator)
    items = []
    for seed in range(BATCH["images"]):
        labels, channels = generate(BATCH["size"], BATCH["count"], seed)
        items.append((labels, _norm01(channels[0]), _norm01(channels[1])))
    results = {}
    for name, fn, arity in _functions():

        def args(batch, arity=arity):
            masks, first, second = (list(group) for group in zip(*batch))
            return (masks, first) if arity == 1 else (first, second, masks)

        probe = _time(fn, args(items[:2]))
        if probe["status"] != "ok":
            results[name] = {"status": "unsupported", "error": probe.get("error")}
            continue
        singles = [
            (labels, a) if arity == 1 else (a, b, labels) for labels, a, b in items
        ]
        results[name] = {
            "status": "ok",
            "loop": _time(lambda: [fn(*one) for one in singles], ()),
            "batch": _time(fn, args(items)),
        }
    Path(out_path).write_text(
        json.dumps(
            {"batch": BATCH, "threads": _threads(), "results": results}, indent=2
        )
    )


def compare_batch(base: dict, head: dict, commit: str = "") -> str:
    """Head's batch vs its own loop of single calls (the thread gain), and head's
    batch vs main's batch (the PR's change). Every supported function is listed."""

    def ms(record, mode):
        timing = record.get(mode, {}) if record.get("status") == "ok" else {}
        reps = timing.get("reps") if timing.get("status") == "ok" else None
        return statistics.median(reps) * 1e3 if reps else None

    def ratio(a, b):
        return f"{a / b:.2f}×" if a and b else "—"

    b = head["batch"]
    ref = f"`{commit[:7]}`" if commit else "PR head"
    out = [
        f"### Batch — {ref}: {b['images']} images of {b['size']}² px, "
        f"{b['count']} objects each, {head['threads']} threads",
        "",
        "`thread gain = loop of single calls / one batch call` (head) · "
        "`vs main = main batch / head batch` (>1 faster) · median of reps · "
        "reported only, not a CI check",
        "",
        "| function | loop (ms) | batch (ms) | thread gain | vs main |",
        "|---|--:|--:|--:|--:|",
    ]
    unsupported = []
    for name in sorted(head["results"]):
        record = head["results"][name]
        if record.get("status") != "ok":
            unsupported.append(name)
            continue
        loop, batch = ms(record, "loop"), ms(record, "batch")
        main = ms(base.get("results", {}).get(name, {}), "batch")
        cells = [f"{loop:.0f}" if loop else "—", f"{batch:.0f}" if batch else "—"]
        out.append(
            f"| `{name}` | "
            + " | ".join(cells + [ratio(loop, batch), ratio(main, batch)])
            + " |"
        )
    if unsupported:
        out += [
            "",
            "_No batch input: " + ", ".join(f"`{n}`" for n in unsupported) + "._",
        ]
    return "\n".join(out)


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


def compare(
    base: dict,
    head: dict,
    commit: str = "",
    base_name: str = "main",
    head_name: str = "head",
    filter_base: dict | None = None,
) -> str:
    groups: dict[tuple, list[str]] = {}
    for e in head["cells"]:
        groups.setdefault((e["size"], e["n_objects"]), []).append(e["key"])
    sizes = sorted({s for s, _ in groups})
    counts = sorted({n for _, n in groups})
    br, hr = base["results"], head["results"]
    # A function is *shown* when it moved vs filter_base (default: the displayed base), but the
    # printed speedup is always base/head. This lets the numba table pick which functions THIS PR
    # changed (filter on numba@main) while displaying the total speedup over numpy (base=numpy@main).
    fr = filter_base["results"] if filter_base else br

    ref = f"`{commit[:7]}`" if commit else "PR head"
    out = [
        f"### Benchmark — {ref}: `{head_name}` vs `{base_name}`",
        "",
        f"`speedup = {base_name}/{head_name}` (>1 faster, <1 slower) · median per cell · "
        f"showing functions that moved ≥{AFFECTED:.2f}× either way",
    ]

    # Historical benchmark JSONs used normalized floats with inferred scale 1.
    # Do not describe the correctness fix as a like-for-like speed regression.
    incompatible_costes = (
        len(
            {
                run.get("costes_scale_max", 1)
                for run in (base, head, filter_base or base)
            }
        )
        > 1
    )
    if incompatible_costes and "costes" in hr:
        out += [
            "",
            "_Costes timing comparison omitted because the threshold grids differ._",
        ]

    lo = 1.0 / AFFECTED  # a cell at or below this is a regression worth reporting
    affected = []  # (function, {(size, count): speedup})
    for fn in sorted(hr):
        if fn == "costes" and incompatible_costes:
            continue
        grid, moves = {}, []
        for size in sizes:
            for n in counts:
                keys = groups.get((size, n), [])
                m = _median_ms(br.get(fn, {}), keys)
                h = _median_ms(hr[fn], keys)
                grid[(size, n)] = (m / h) if (m and h) else None
                f = _median_ms(fr.get(fn, {}), keys)
                if f and h:
                    moves.append(f / h)  # movement vs filter_base decides inclusion
        if moves and (max(moves) >= AFFECTED or min(moves) <= lo):
            affected.append((fn, grid))

    if not affected:
        out += ["", f"_No function moved by ≥{AFFECTED:.2f}× (faster or slower)._"]
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
    r.add_argument("--accelerator", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("--head", required=True)
    c.add_argument("--commit", default="")
    c.add_argument("--base-name", default="main")
    c.add_argument("--head-name", default="head")
    c.add_argument("--filter-base", default=None)
    c.add_argument("--md")
    rb = sub.add_parser("run-batch")
    rb.add_argument("--out", required=True)
    rb.add_argument("--accelerator", default=None)
    cb = sub.add_parser("compare-batch")
    cb.add_argument("--base", required=True)
    cb.add_argument("--head", required=True)
    cb.add_argument("--commit", default="")
    cb.add_argument("--md")
    a = p.parse_args(argv)
    if a.cmd == "run":
        run(a.out, a.accelerator)
    elif a.cmd == "run-batch":
        run_batch(a.out, a.accelerator)
    elif a.cmd == "compare-batch":
        md = compare_batch(
            json.loads(Path(a.base).read_text()),
            json.loads(Path(a.head).read_text()),
            a.commit,
        )
        (Path(a.md).write_text(md) if a.md else print(md))
    else:
        md = compare(
            json.loads(Path(a.base).read_text()),
            json.loads(Path(a.head).read_text()),
            a.commit,
            a.base_name,
            a.head_name,
            json.loads(Path(a.filter_base).read_text()) if a.filter_base else None,
        )
        (Path(a.md).write_text(md) if a.md else print(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
