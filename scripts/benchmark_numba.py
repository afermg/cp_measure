"""Benchmark Numba-accelerated modules against their Python fallbacks.

Outputs a Markdown table suitable for GitHub Actions job summaries.
Always exits 0 (non-blocking) — import errors and exceptions are reported
but never fail the build.

Usage:
    uv run python scripts/benchmark_numba.py
"""

import importlib
import statistics
import sys
import time
import traceback

import numpy as np


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _make_elliptical_objects(height=540, width=540, n_objects=50, seed=0):
    """Create non-overlapping elliptical objects for benchmarking.

    Returns ``(masks, pixels, actual_count)`` where *actual_count* is the
    number of objects that were successfully placed.
    """
    rng = np.random.RandomState(seed)
    masks = np.zeros((height, width), dtype=np.int32)
    pixels = rng.random((height, width)).astype(np.float64)

    placed = 0
    attempts = 0
    while placed < n_objects and attempts < n_objects * 20:
        attempts += 1
        ry = rng.randint(5, 20)
        rx = rng.randint(5, 20)
        cy = rng.randint(ry + 1, height - ry - 1)
        cx = rng.randint(rx + 1, width - rx - 1)

        y_lo = max(0, cy - ry)
        y_hi = min(height, cy + ry + 1)
        x_lo = max(0, cx - rx)
        x_hi = min(width, cx + rx + 1)
        if np.any(masks[y_lo:y_hi, x_lo:x_hi] > 0):
            continue

        for y in range(y_lo, y_hi):
            for x in range(x_lo, x_hi):
                if ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0:
                    masks[y, x] = placed + 1
        placed += 1

    return masks, pixels, placed


# ---------------------------------------------------------------------------
# Benchmark scenarios — three sizes per module
# ---------------------------------------------------------------------------

INTENSITY_SCENARIOS = [
    {"height": 256, "width": 256, "n_objects": 10, "label": "small"},
    {"height": 540, "width": 540, "n_objects": 50, "label": "medium"},
    {"height": 1080, "width": 1080, "n_objects": 150, "label": "large"},
]


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "intensity": {
        "numba": ("cp_measure.core._intensity_numba", "get_intensity_numba"),
        "python": (
            "cp_measure.core.measureobjectintensity",
            "_get_intensity_python",
        ),
        "scenarios": INTENSITY_SCENARIOS,
    },
    # Future: "texture", "correlation", etc.
}

N_RUNS = 3  # runs per implementation (median is reported)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _import_func(module_path, func_name):
    """Import and return a callable, or None on failure."""
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _bench(func, args, n_runs):
    """Return the median wall-clock time (seconds) over *n_runs*."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    rows = []

    for name, cfg in BENCHMARKS.items():
        numba_mod, numba_fn = cfg["numba"]
        python_mod, python_fn = cfg["python"]

        # Import Numba function (skip if unavailable)
        try:
            fn_numba = _import_func(numba_mod, numba_fn)
        except Exception:
            print(
                f"  note: skipping '{name}' — could not import Numba path",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            continue

        # Import Python fallback
        try:
            fn_python = _import_func(python_mod, python_fn)
        except Exception:
            print(
                f"  note: skipping '{name}' — could not import Python path",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            continue

        # Warm up JIT on small data before timing
        warm_masks, warm_pixels, _ = _make_elliptical_objects(
            height=64,
            width=64,
            n_objects=3,
        )
        fn_numba(warm_masks, warm_pixels)

        # Run each scenario
        for scenario in cfg["scenarios"]:
            masks, pixels, n_objects = _make_elliptical_objects(
                height=scenario["height"],
                width=scenario["width"],
                n_objects=scenario["n_objects"],
            )

            t_python = _bench(fn_python, (masks, pixels), N_RUNS)
            t_numba = _bench(fn_numba, (masks, pixels), N_RUNS)
            speedup = t_python / t_numba if t_numba > 0 else float("inf")

            h, w = masks.shape
            rows.append(
                (
                    name,
                    scenario["label"],
                    n_objects,
                    f"{h}\u00d7{w}",
                    t_python,
                    t_numba,
                    speedup,
                )
            )

    # Output
    if not rows:
        print("No Numba modules available — nothing to benchmark.")
        return

    # Format cells, then pad each column to uniform width
    headers = [
        "Module",
        "Scenario",
        "Objects",
        "Image",
        "Python (s)",
        "Numba (s)",
        "Speedup",
    ]
    formatted = []
    for name, scenario, n_obj, img, t_py, t_nb, spd in rows:
        formatted.append(
            [
                name,
                scenario,
                str(n_obj),
                img,
                f"{t_py:.3f}",
                f"{t_nb:.3f}",
                f"{spd:.0f}\u00d7",
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in formatted:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(cells):
        padded = [cell.ljust(col_widths[i]) for i, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    print("## Numba Benchmark Results\n")
    print(_fmt_row(headers))
    print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
    for row in formatted:
        print(_fmt_row(row))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Benchmark script failed (non-blocking):", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    sys.exit(0)
