"""Time every public ``get_*`` function over a fixture matrix → JSON (run per env; compare.py diffs two).

Functions come from the live ``bulk`` registry at HEAD (a PR-added feature is timed, reported "new").
Each channel is normalised to ``[0, 1]`` (``get_texture`` requires it); calls are positional per arity:
core ``fn(labels, img)``, correlation ``fn(img1, img2, labels)``.
"""

from __future__ import annotations

# Thread pinning MUST precede numpy/BLAS import: single-thread timing is representative because the
# repo forbids parallelism inside feature functions, and it removes contention noise. The CI driver
# also sets these; we set them defensively here before any numpy import.
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import statistics  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter  # noqa: E402

from cp_measure._bench import fixtures  # noqa: E402

DEFAULT_WARMUP = 1
DEFAULT_REPS = 5
DEFAULT_TIMEOUT = 120.0


@dataclass
class Func:
    label: str
    fn: object
    arity: int
    kwargs: dict


class _Timeout(Exception):
    pass


@contextmanager
def _time_limit(seconds: float):
    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _has_legacy(fn) -> bool:
    fn = getattr(fn, "func", fn)  # unwrap functools.partial
    try:
        return "legacy" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def enumerate_functions() -> list[Func]:
    """All public get_* functions from the live registries, plus a [legacy] variant where supported."""
    from cp_measure import bulk

    out: list[Func] = []
    for arity, registry in (
        (1, bulk.get_core_measurements()),
        (2, bulk.get_correlation_measurements()),
    ):
        for name, fn in registry.items():
            out.append(Func(name, fn, arity, {}))
            # [legacy] reuses the same fn, so for a numba backend it shares the base variant's
            # JIT-warmed code — read the legacy-vs-base delta with that caveat.
            if _has_legacy(fn):
                out.append(Func(f"{name}[legacy]", fn, arity, {"legacy": True}))
    return out


def _norm01(image):
    image = image.astype("float64")
    lo, hi = float(image.min()), float(image.max())
    return (image - lo) / (hi - lo) if hi > lo else image - lo  # constant image → zeros


def _call_args(func: Func, labels, channels):
    if func.arity == 1:
        return (labels, _norm01(channels[0]))
    return (_norm01(channels[0]), _norm01(channels[1]), labels)


def time_call(func: Func, args, warmup: int, reps: int, timeout: float) -> dict:
    try:
        with _time_limit(timeout):
            for _ in range(warmup):
                func.fn(*args, **func.kwargs)
            times = []
            for _ in range(reps):
                t0 = perf_counter()
                func.fn(*args, **func.kwargs)
                times.append(perf_counter() - t0)
    except _Timeout:
        return {"status": "timeout", "timeout_s": timeout}
    except (
        Exception
    ) as exc:  # a function that can't handle the synthetic input — recorded, not fatal
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}
    return {
        "status": "ok",
        "min": min(times),
        "median": statistics.median(times),
        "reps": times,
    }


def run(
    fixtures_dir,
    out_path,
    warmup=DEFAULT_WARMUP,
    reps=DEFAULT_REPS,
    timeout=DEFAULT_TIMEOUT,
):
    manifest = fixtures.load_manifest(fixtures_dir)
    funcs = enumerate_functions()
    results: dict[str, dict] = {f.label: {} for f in funcs}
    # Outer loop over fixtures so each .npz is loaded (and sha-verified) once.
    for entry in manifest["fixtures"]:
        labels, channels = fixtures.load_fixture(fixtures_dir, entry)
        for func in funcs:
            try:
                args = _call_args(func, labels, channels)
            except Exception as exc:  # e.g. correlation fn on a 1-channel fixture
                results[func.label][entry["key"]] = {
                    "status": "error",
                    "error": str(exc)[:200],
                }
                continue
            results[func.label][entry["key"]] = time_call(
                func, args, warmup, reps, timeout
            )

    out = {
        "meta": {
            "synth_version": manifest["synth_version"],
            "matrix": manifest["matrix"],
            "n_fixtures": len(manifest["fixtures"]),
            "n_functions": len(funcs),
            "warmup": warmup,
            "reps": reps,
            "timeout_s": timeout,
            "threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "fixtures": manifest["fixtures"],
        "results": results,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Time all get_* functions over a fixture matrix."
    )
    p.add_argument(
        "--fixtures", required=True, help="fixtures dir (with manifest.json)"
    )
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    a = p.parse_args(argv)
    out = run(a.fixtures, a.out, a.warmup, a.reps, a.timeout)
    ok = sum(
        1 for fn in out["results"].values() for c in fn.values() if c["status"] == "ok"
    )
    total = sum(len(c) for c in out["results"].values())
    print(
        f"timed {out['meta']['n_functions']} functions over {len(out['fixtures'])} fixtures "
        f"({ok}/{total} cells ok) -> {a.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
