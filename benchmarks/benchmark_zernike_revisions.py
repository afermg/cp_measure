"""Reproduce the PR58 Zernike comparison without installing historical packages.

Run with the project's Python environment (including the numba extra):
    python benchmarks/benchmark_zernike_revisions.py --cpu 86 --out /tmp/zernike-run

Three rotated, sequential process rounds; five timed calls after one excluded
warmup per feature/scene/process. All revisions use one interpreter, dependency
set, CPU, and thread limit. Statistics are median [p25, p75], NOT confidence
intervals. Old = v0.1.19; current = main after PR115; candidate defaults to HEAD.
Sources are extracted read-only from Git into the output directory. No installs.
"""

import argparse
import csv
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tarfile
import time
import warnings
from xml.sax.saxutils import escape

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
for var in THREAD_VARS:
    os.environ[var] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

SCENES = [(256, 16), (512, 64), (1024, 256)]
FEATURES = ["zernike", "radial_zernikes"]
VARIANTS = ["old-numpy", "current-numpy", "candidate-numba"]
SEED = 20261020


def generate(size, n):
    """Exactly the same seeded elliptical scenes as the earlier version comparison."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[:size, :size]
    labels = np.zeros((size, size), np.int32)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    for k in range(n):
        r, c = divmod(k, cols)
        cy = (r + 0.5) * size / rows + (0.5 if k % 3 == 0 else 0)
        cx = (c + 0.5) * size / cols + (0.5 if k % 3 == 0 else 0)
        a = (0.27 + 0.06 * rng.random()) * size / rows
        b = (0.27 + 0.06 * rng.random()) * size / cols
        labels[((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2 <= 1] = k + 1
    img = np.zeros((size, size), np.float64)
    for _ in range(5):
        cy, cx = rng.uniform(0, size, 2)
        s = rng.uniform(size / 10, size / 5)
        img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    img += 0.08 * rng.random(img.shape) + 0.06 * (1 + np.sin(xx / 3) * np.cos(yy / 5))
    img = 0.05 + 0.9 * (img - img.min()) / (img.max() - img.min())
    return labels, np.ascontiguousarray(img)


def digest(*arrays):
    return hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()


def worker(args):
    # v0.1.19 is a namespace package: PYTHONPATH alone would allow an installed
    # regular package to override it. Load only this exact extracted source root.
    src = args.src.resolve()
    spec = importlib.machinery.PathFinder.find_spec("cp_measure", [str(src)])
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["cp_measure"] = module
    if spec.loader is not None:
        spec.loader.exec_module(module)
    from cp_measure import bulk

    assert Path(bulk.__file__).resolve().is_relative_to(src)
    if hasattr(module, "set_accelerator"):
        module.set_accelerator("numba" if args.worker.endswith("numba") else None)
    funcs = bulk.get_core_measurements()  # public default, including sanitation
    versions = {
        k: importlib.metadata.version(k)
        for k in (
            "numpy",
            "scipy",
            "scikit-image",
            "centrosome",
            "mahotas",
            "numba",
            "llvmlite",
        )
    }
    result = dict(
        variant=args.worker,
        python=sys.version,
        versions=versions,
        affinity=sorted(os.sched_getaffinity(0)),
        bulk_file=bulk.__file__,
        threads={k: os.environ[k] for k in THREAD_VARS},
        cases=[],
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for size, n in SCENES:
        labels, pixels = generate(size, n)
        before = digest(labels, pixels)
        for feature in FEATURES:
            params = {
                "zernike_numbers" if feature == "zernike" else "zernike_degree": 9
            }
            fn = funcs[feature]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                reference = fn(labels, pixels, **params)
                warmup = time.perf_counter() - start
                durations = []
                for _ in range(5):
                    start = time.perf_counter()
                    measured = fn(labels, pixels, **params)
                    durations.append(time.perf_counter() - start)
            assert list(reference) == list(measured)
            for key in reference:
                # NumPy's transcendental SIMD paths may differ by a few ulps
                # between allocations; require tight numerical, not bitwise,
                # repeatability (much stricter than the backend contract).
                np.testing.assert_allclose(
                    reference[key],
                    measured[key],
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )
            assert digest(labels, pixels) == before
            np.savez_compressed(args.out / f"{size}-{feature}.npz", **measured)
            case = dict(
                size=size,
                objects=n,
                feature=feature,
                params=params,
                input_sha256=before,
                warmup_seconds=warmup,
                durations_seconds=durations,
                warnings=sorted({str(w.message) for w in caught}),
                keys=list(measured),
                inputs_unchanged=True,
            )
            result["cases"].append(case)
            print(
                args.worker,
                size,
                feature,
                f"{statistics.median(durations) * 1000:.3f} ms",
                flush=True,
            )
    (args.out / "timings.json").write_text(json.dumps(result, indent=2) + "\n")


def plot(rows, out):
    colors = ["#0072b2", "#d55e00", "#009e73"]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="710" viewBox="0 0 1120 710">',
        '<rect width="1120" height="710" fill="white"/>',
        '<g font-family="sans-serif" fill="#222">',
        '<text x="24" y="30" font-size="22">Rebased PR58: median runtime and 25th–75th percentiles</text>',
        '<text x="24" y="55" font-size="14">15 calls / cell; warmup excluded; descriptive IQR, not confidence intervals</text>',
    ]
    for j, name in enumerate(VARIANTS):
        svg.append(
            f'<text x="{24 + j * 355}" y="85" fill="{colors[j]}" font-size="16">{escape(name)}</text>'
        )
    for f, feature in enumerate(FEATURES):
        subset = [r for r in rows if r["feature"] == feature]
        lo = math.floor(math.log10(min(r["p25_ms"] for r in subset)))
        hi = math.ceil(math.log10(max(r["p75_ms"] for r in subset)))
        left = 145 + 550 * f
        width = 380

        def xpos(value):
            return left + width * (math.log10(value) - lo) / (hi - lo)

        svg.append(f'<text x="{left}" y="120" font-size="20">{feature}</text>')
        for exp in range(lo, hi + 1):
            x = xpos(10**exp)
            svg += [
                f'<path d="M{x} 135 V640" stroke="#ddd"/>',
                f'<text x="{x}" y="665" text-anchor="middle">{10**exp:g}</text>',
            ]
        for s, (size, n) in enumerate(SCENES):
            y = 160 + s * 165
            svg.append(
                f'<text x="{left - 125}" y="{y}" font-size="14">{size}² / {n} obj</text>'
            )
            for j, variant in enumerate(VARIANTS):
                row = next(
                    r for r in subset if r["size"] == size and r["variant"] == variant
                )
                yy = y + 25 + j * 35
                x1, x2, xm = (xpos(row[k]) for k in ("p25_ms", "p75_ms", "median_ms"))
                svg.append(
                    f'<path d="M{x1} {yy - 5} V{yy + 5} M{x1} {yy} H{x2} M{x2} {yy - 5} V{yy + 5}" stroke="{colors[j]}" stroke-width="2"/>'
                )
                svg.append(f'<circle cx="{xm}" cy="{yy}" r="4" fill="{colors[j]}"/>')
        svg.append(
            f'<text x="{left + width / 2}" y="695" text-anchor="middle">milliseconds (log scale)</text>'
        )
    (out / "iqr.svg").write_text("\n".join(svg) + "</g></svg>\n")


def summarize(out, manifest):
    rows, validation = [], []
    dependencies, hashes = set(), {}
    for size, n in SCENES:
        for feature in FEATURES:
            with np.load(
                out / "rounds/1/current-numpy" / f"{size}-{feature}.npz"
            ) as archive:
                ref = {k: archive[k] for k in archive.files}
            for variant in VARIANTS:
                durations, medians, max_diff = [], [], 0.0
                for rnd in range(1, 4):
                    root = out / f"rounds/{rnd}" / variant
                    record = json.loads((root / "timings.json").read_text())
                    dependencies.add(json.dumps(record["versions"], sort_keys=True))
                    assert record["affinity"] == [manifest["cpu"]]
                    assert all(v == "1" for v in record["threads"].values())
                    case = next(
                        c
                        for c in record["cases"]
                        if c["size"] == size and c["feature"] == feature
                    )
                    hashes.setdefault(size, set()).add(case["input_sha256"])
                    assert (
                        case["inputs_unchanged"] and len(case["durations_seconds"]) == 5
                    )
                    durations.extend(case["durations_seconds"])
                    medians.append(statistics.median(case["durations_seconds"]))
                    with np.load(root / f"{size}-{feature}.npz") as got:
                        assert list(ref) == got.files
                        for key in ref:
                            assert ref[key].shape == got[key].shape == (n,)
                            assert ref[key].dtype == got[key].dtype
                            np.testing.assert_allclose(
                                got[key],
                                ref[key],
                                rtol=1e-6,
                                atol=1e-8,
                                equal_nan=True,
                                err_msg=f"{variant}/{size}/{key}",
                            )
                            finite = np.isfinite(ref[key]) & np.isfinite(got[key])
                            if finite.any():
                                max_diff = max(
                                    max_diff,
                                    float(
                                        np.max(
                                            np.abs(got[key][finite] - ref[key][finite])
                                        )
                                    ),
                                )
                q25, med, q75 = np.quantile(durations, [0.25, 0.5, 0.75]) * 1000
                rows.append(
                    dict(
                        size=size,
                        objects=n,
                        feature=feature,
                        variant=variant,
                        n=15,
                        median_ms=med,
                        p25_ms=q25,
                        p75_ms=q75,
                        round_spread_ratio=max(medians) / min(medians),
                    )
                )
                validation.append(
                    dict(
                        size=size,
                        feature=feature,
                        variant=variant,
                        keys=len(ref),
                        max_abs_diff=max_diff,
                        passed=True,
                    )
                )
    assert len(dependencies) == 1 and all(len(v) == 1 for v in hashes.values())
    with (out / "statistics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = dict(
        dependencies=json.loads(next(iter(dependencies))),
        input_hashes={s: next(iter(v)) for s, v in hashes.items()},
        tolerance=dict(rtol=1e-6, atol=1e-8),
        comparisons=validation,
        max_round_spread_ratio=max(r["round_spread_ratio"] for r in rows),
    )
    (out / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "# Rebased PR58 benchmark",
        "",
        "Milliseconds: median [p25, p75]; 15 timed calls per cell, warmup/JIT excluded.",
        "",
        "| Scene | Feature | v0.1.19 NumPy | Current-main NumPy | Rebased Numba | Old/Numba | Current/Numba |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for size, n in SCENES:
        for feature in FEATURES:
            cells = [
                next(
                    r
                    for r in rows
                    if r["size"] == size
                    and r["feature"] == feature
                    and r["variant"] == v
                )
                for v in VARIANTS
            ]
            vals = [
                f"{r['median_ms']:.2f} [{r['p25_ms']:.2f}, {r['p75_ms']:.2f}]"
                for r in cells
            ]
            ratios = [r["median_ms"] / cells[2]["median_ms"] for r in cells[:2]]
            lines.append(
                f"| {size}² / {n} | {feature} | "
                + " | ".join(vals)
                + f" | {ratios[0]:.2f}× | {ratios[1]:.2f}× |"
            )
    lines += [
        "",
        "## Method and limits",
        "",
        f"- Source revisions: `{manifest['refs']}`.",
        f"- Python: {platform.python_version()}; dependencies: `{audit['dependencies']}`.",
        f"- CPU {manifest['cpu']}, all thread limits = 1; one host/core for all sizes and revisions.",
        "- Three rotated sequential process rounds × five calls; one excluded warmup per case/process. Public bulk registry defaults, degree 9.",
        "- Same seed (20261020), inputs, float64 pixels, int32 masks and density as the previous comparison. All variants freshly rerun; no timing ratios across different sessions.",
        "- Input hashes, key order, shapes, dtypes, nonfinite patterns and numerical tolerance checked against current NumPy in every round. Inputs unchanged.",
        "- IQR describes timing variability, not confidence. Speedup ranges are across workload conditions; ratios have no inferred confidence interval.",
        f"- Largest round-median max/min ratio: {audit['max_round_spread_ratio']:.3f}; CPU affinity does not reserve a shared host.",
        "- Function timings only: no end-to-end pipeline, cold-start or memory claim. Historical comparison combines version and backend improvements.",
        "- Radial azimuthal powers remain in NumPy to preserve cancellation-sensitive phase; Horner evaluation and reductions are fused in Numba.",
        "",
        "[Error-bar plot](iqr.svg) · [Statistics](statistics.csv) · [Validation](validation.json) · [Raw timings](raw-timings.json)",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))
    raw = [
        dict(round=r, **json.loads((out / f"rounds/{r}/{v}/timings.json").read_text()))
        for r in range(1, 4)
        for v in VARIANTS
    ]
    (out / "raw-timings.json").write_text(json.dumps(raw, indent=2) + "\n")
    plot(rows, out)
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--old-ref", default="54e5c094e4a2c9079f747bdc419eb5623ad2d4d6")
    parser.add_argument(
        "--current-ref", default="c698d440e6b669222efdf0d128dad88843bf2677"
    )
    parser.add_argument("--candidate-ref", default="HEAD")
    parser.add_argument("--worker", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--src", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.cpu is None or args.cpu not in os.sched_getaffinity(0):
        parser.error("--cpu must be an available CPU; select one before the run")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("--out must be empty (preserve previous raw data)")
    refs = dict(zip(VARIANTS, [args.old_ref, args.current_ref, args.candidate_ref]))
    refs = {
        k: subprocess.check_output(["git", "rev-parse", v], text=True).strip()
        for k, v in refs.items()
    }
    manifest = dict(
        refs=refs, cpu=args.cpu, started_unix=time.time(), platform=platform.platform()
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for variant, ref in refs.items():
        archive = subprocess.check_output(["git", "archive", ref, "src"])
        target = out / "sources" / variant
        target.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(target, filter="data")
    for rnd in range(3):
        for variant in VARIANTS[rnd:] + VARIANTS[:rnd]:
            env = dict(os.environ, NUMBA_CACHE_DIR=str(out / "numba-cache" / str(rnd)))
            cmd = [
                "taskset",
                "-c",
                str(args.cpu),
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                variant,
                "--src",
                str(out / "sources" / variant / "src"),
                "--out",
                str(out / f"rounds/{rnd + 1}" / variant),
            ]
            subprocess.run(cmd, env=env, check=True, timeout=300)
    manifest["finished_unix"] = time.time()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summarize(out, manifest)


if __name__ == "__main__":
    main()
