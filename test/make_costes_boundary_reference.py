"""Generate boundary fixtures from the same pinned CP methods as normal fixtures.

Usage: python test/make_costes_boundary_reference.py --source UPSTREAM.py --out NEW.json
Not run by pytest. Only the two inspected, hash-verified threshold ASTs execute.
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

from make_costes_reference import COMMIT, SOURCE_SHA256, URL


SAMPLES = {
    "two_pixel_background": (
        [2, 3, 10, 12, 20, 30, 31, 34],
        [4, 7, 9, 16, 17, 33, 28, 39],
    ),
    "one_pixel_background": (
        [9, 23, 12, 18, 14, 21, 10, 3],
        [9, 19, 19, 29, 8, 13, 28, 4],
    ),
    "constant_background": (
        [0] * 12 + [1, 2, 3, 4, 5, 6],
        [0] * 12 + [1, 3, 2, 5, 4, 6],
    ),
    "no_nonnegative_candidate": (
        [25, 19, 15, 8, 9, 2, 3, 1],
        [6, 24, 19, 27, 15, 18, 29, 22],
    ),
    "anticorrelated_endpoints": ([0, 255, 0, 255, 0, 255], [255, 0, 255, 0, 255, 0]),
}


def generate(source, output):
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "MeasureColocalization"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("linear_costes", "bisection_costes")
    ]
    assert len(methods) == 2
    namespace = {"numpy": np, "scipy": scipy, "M_ACCURATE": "Accurate"}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    inputs = [
        (name, np.asarray(x) / 255, np.asarray(y) / 255, scale)
        for name, (x, y) in SAMPLES.items()
        for scale in (255, 65535)
    ]
    inputs.append(
        ("empty_background_coarse_grid", np.arange(1, 7) / 10, np.arange(1, 7) / 10, 2)
    )
    cases = []
    for name, first, second, scale in inputs:
        for mode in ("Faster", "Fast", "Accurate"):
            owner = SimpleNamespace(fast_costes=SimpleNamespace(value=mode))
            search = namespace[
                "bisection_costes" if mode == "Faster" else "linear_costes"
            ]
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                t1, t2 = search(owner, first, second, scale)
            selected = (first < t1) | (second < t2)
            joint = (first > t1) & (second > t2)
            d1, d2 = first[first >= t1].sum(), second[second >= t2].sum()
            # Preserve CPM's independent zero-denominator contract (both NaN).
            fractions = (
                [None, None]
                if d1 <= 0 or d2 <= 0
                else [float(first[joint].sum() / d1), float(second[joint].sum() / d2)]
            )
            cases.append(
                {
                    "name": name,
                    "mode": mode,
                    "scale_max": scale,
                    "first": first.tolist(),
                    "second": second.tolist(),
                    "thresholds": [float(t1), float(t2)],
                    "fractions": fractions,
                    "background_count": int(selected.sum()),
                    "background_constant": bool(
                        selected.any()
                        and (
                            np.ptp(first[selected]) == 0
                            or np.ptp(second[selected]) == 0
                        )
                    ),
                }
            )
    result = {
        "upstream_commit": COMMIT,
        "source_url": URL,
        "source_sha256": SOURCE_SHA256,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scope": "Unmodified CP threshold methods; fractions use CPM's strict numerator/inclusive denominator and retained zero-denominator policy. Not full pipeline equivalence. Sample names describe Faster-mode cases; linear outcomes can differ.",
        "cases": cases,
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{len(cases)} boundary fixtures -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.out)
