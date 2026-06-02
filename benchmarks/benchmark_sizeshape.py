"""
Compare old vs new sizeshape implementations for:
  - numerical equivalence (per-feature, per-label)
  - runtime

The OLD implementation is fetched from a git ref (default: main) by reading
both `measureobjectsizeshape.py` and `utils.py`, since the speedup touches
both. The NEW implementation is loaded from src/.

Usage
-----
    python benchmarks/benchmark_sizeshape.py [--ref main]
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SIZESHAPE_REL = "src/cp_measure/core/measureobjectsizeshape.py"
UTILS_REL = "src/cp_measure/utils.py"


def _git_show(ref: str, rel_path: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{ref}:{rel_path}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def _load_module(name: str, source: str):
    """Compile ``source`` into a module named ``name`` and register it."""
    tmp = Path(tempfile.mkstemp(prefix=f"{name}_", suffix=".py")[1])
    tmp.write_text(source)
    spec = importlib.util.spec_from_file_location(name, str(tmp))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_old_from_ref(ref: str):
    """Materialize a cp_measure_old package from ``ref`` and import sizeshape from it."""
    utils_src = _git_show(ref, UTILS_REL)
    sizeshape_src = _git_show(ref, SIZESHAPE_REL).replace(
        "from cp_measure.utils import", "from cp_measure_old.utils import"
    )

    pkg_dir = Path(tempfile.mkdtemp(prefix="cp_measure_old_"))
    (pkg_dir / "cp_measure_old").mkdir()
    (pkg_dir / "cp_measure_old" / "__init__.py").write_text("")
    (pkg_dir / "cp_measure_old" / "utils.py").write_text(utils_src)
    (pkg_dir / "cp_measure_old" / "core").mkdir()
    (pkg_dir / "cp_measure_old" / "core" / "__init__.py").write_text("")
    (pkg_dir / "cp_measure_old" / "core" / "measureobjectsizeshape.py").write_text(
        sizeshape_src
    )

    sys.path.insert(0, str(pkg_dir))
    import importlib

    return importlib.import_module("cp_measure_old.core.measureobjectsizeshape")


def make_data(size: int = 1024, n_labels: int = 64, seed: int = 42):
    rng = np.random.default_rng(seed)
    pixels = rng.random((size, size), dtype=np.float32)

    labels = np.zeros((size, size), dtype=np.int32)
    side = max(8, size // (int(np.sqrt(n_labels)) + 2))
    label_id = 1
    for r in range(0, size - side, side + 4):
        for c in range(0, size - side, side + 4):
            if label_id > n_labels:
                break
            h = side + rng.integers(-2, 3)
            w = side + rng.integers(-2, 3)
            labels[r : r + h, c : c + w] = label_id
            label_id += 1
        if label_id > n_labels:
            break
    return pixels, labels


FUNCS = ["get_sizeshape", "get_feret", "get_zernike"]


def time_func(fn, *args, repeat: int = 3, **kwargs):
    out = fn(*args, **kwargs)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return out, best


def compare_results(old: dict, new: dict, name: str) -> bool:
    ok = True
    keys = sorted(set(old) | set(new))
    for k in keys:
        if k not in old:
            print(f"  {name}: missing key in OLD: {k}")
            ok = False
            continue
        if k not in new:
            print(f"  {name}: missing key in NEW: {k}")
            ok = False
            continue
        ov = np.asarray(old[k], dtype=float)
        nv = np.asarray(new[k], dtype=float)
        if ov.shape != nv.shape:
            print(f"  {name}: shape mismatch {k}: old={ov.shape} new={nv.shape}")
            ok = False
            continue
        mask = ~(np.isnan(ov) & np.isnan(nv))
        if not np.allclose(ov[mask], nv[mask], rtol=1e-5, atol=1e-7, equal_nan=False):
            diff = np.abs(ov[mask] - nv[mask])
            print(
                f"  {name}: {k}  max|diff|={diff.max():.3e}  "
                f"mean|diff|={diff.mean():.3e}  N={ov.size}"
            )
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="main", help="Git ref to compare against.")
    args = parser.parse_args()

    OLD = _load_old_from_ref(args.ref)
    # Import NEW via the installed package
    from cp_measure.core import measureobjectsizeshape as NEW

    sizes = [
        (256, 16),
        (512, 64),
        (1024, 128),
    ]
    print(f"Comparing NEW (src/) against ref={args.ref}")
    for size, n_labels in sizes:
        print(f"\n=== size={size}x{size}  n_labels={n_labels} ===")
        pixels, labels = make_data(size=size, n_labels=n_labels)

        for fn_name in FUNCS:
            old_fn = getattr(OLD, fn_name)
            new_fn = getattr(NEW, fn_name)

            old_out, t_old = time_func(old_fn, labels, pixels)
            new_out, t_new = time_func(new_fn, labels, pixels)

            equal = compare_results(old_out, new_out, fn_name)
            tag = "OK " if equal else "FAIL"
            speedup = t_old / t_new if t_new > 0 else float("inf")
            print(
                f"  [{tag}] {fn_name:20s} "
                f"old={t_old * 1e3:8.2f}ms  new={t_new * 1e3:8.2f}ms  "
                f"speedup={speedup:5.2f}x"
            )


if __name__ == "__main__":
    main()
