"""Regenerate small fixtures from pinned, unmodified CellProfiler threshold methods.

Not run by pytest. Download the documented upstream file, then pass --source.
Only two inspected method ASTs are executed; no CellProfiler application imports.
This validates threshold-routine parity, not full CellProfiler pipeline parity.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import scipy.stats

COMMIT = "450abdc2eaa0332cb6d1d4aaed4bf0a4b843368d"
SOURCE_SHA256 = "4dedd8faf2ff1fc78f1777be517c7318420293f691e13de81888dd92b2e677f7"
URL = f"https://raw.githubusercontent.com/CellProfiler/CellProfiler/{COMMIT}/src/frontend/cellprofiler/modules/measurecolocalization.py"


def generate(source, output):
    text = source.read_text()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    tree = ast.parse(text)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MeasureColocalization"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ["linear_costes", "bisection_costes"]
    ]
    assert len(methods) == 2
    namespace = {"numpy": np, "scipy": scipy, "M_ACCURATE": "Accurate"}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    cases = []
    for seed in [3, 8, 17]:
        rng = np.random.default_rng(seed)
        first = rng.integers(0, 12, 128)
        second = rng.integers(0, 12, 128)
        first[64:] = rng.integers(50, 180, 64)
        second[64:] = (first[64:] * 0.7 + rng.integers(0, 20, 64)).astype(int)
        for scale in [255, 65535]:
            multiplier = scale // 255
            raw_a, raw_b = first * multiplier, second * multiplier
            a, b = raw_a / scale, raw_b / scale
            for mode in ["Faster", "Fast", "Accurate"]:
                owner = SimpleNamespace(fast_costes=SimpleNamespace(value=mode))
                method = "bisection_costes" if mode == "Faster" else "linear_costes"
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    t1, t2 = namespace[method](owner, a, b, scale)
                below = (a < t1) | (b < t2)
                if (
                    caught
                    or below.sum() < 3
                    or np.ptp(a[below]) == 0
                    or np.ptp(b[below]) == 0
                ):
                    raise RuntimeError(
                        f"Reference fixture is degenerate: {seed}, {scale}, {mode}"
                    )
                joint = (a > t1) & (b > t2)
                c1, c2 = (
                    a[joint].sum() / a[a >= t1].sum(),
                    b[joint].sum() / b[b >= t2].sum(),
                )
                cases.append(
                    {
                        "seed": seed,
                        "scale_max": scale,
                        "mode": mode,
                        "raw_first": raw_a.tolist(),
                        "raw_second": raw_b.tolist(),
                        "thresholds": [float(t1), float(t2)],
                        "fractions": [float(c1), float(c2)],
                    }
                )
    result = {
        "upstream_commit": COMMIT,
        "source_url": URL,
        "source_sha256": SOURCE_SHA256,
        "scope": "unmodified threshold methods; fractions follow the documented strict-numerator/inclusive-denominator formula; not the full application",
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{len(cases)} upstream reference cases -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.out)
