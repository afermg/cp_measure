#!/usr/bin/env python3
"""Small seeded featurizer benchmark; correctness gates, informational timings.

Run the same script in isolated head/main environments, then compare their JSON
and NPZ artifacts. Includes default pair selection and an ordered compatibility
control, on default and CP-matched colocalization settings. No CellProfiler run.
"""

import argparse
import hashlib
import inspect
import json
import platform
import statistics
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

CHANNELS = ["AGP", "DNA", "ER", "Mito", "RNA"]
FEATURES = ("pearson", "costes", "manders_fold", "rwc")
REPS = 3


def inputs():
    rng = np.random.default_rng(177)
    common = rng.integers(20, 180, size=(64, 64))
    image = np.stack(
        [common + rng.integers(0, 55, size=common.shape) for _ in CHANNELS]
    ).astype(np.uint8)[None]
    masks = np.zeros((1, 1, 64, 64), dtype=np.int32)
    for label, (y, x) in enumerate(((4, 4), (4, 36), (36, 4), (36, 36)), 1):
        masks[0, 0, y : y + 20, x : x + 20] = label
    return image, masks


def configuration(policy, mode):
    from cp_measure.featurizer import make_featurizer_config

    flags = dict.fromkeys(
        (
            "intensity",
            "texture",
            "granularity",
            "radial_distribution",
            "radial_zernikes",
            "sizeshape",
            "zernike",
            "feret",
        ),
        False,
    )
    flags.update({f"correlation_{feature}": True for feature in FEATURES})
    flags["objects"] = ["cells"]
    flags["correlation_costes_params"] = {
        "scale_max": 255,
        "scope": "image" if policy == "matched" else "object",
        "precision": "input" if policy == "matched" else "float64",
    }
    flags["correlation_rwc_params"] = {
        "thr": 15,
        "rank_scope": "object_set" if policy == "matched" else "object",
    }
    flags["correlation_manders_fold_params"] = {"thr": 15}
    supports_mode = (
        "costes_pair_mode" in inspect.signature(make_featurizer_config).parameters
    )
    if mode == "ordered" and supports_mode:
        flags["costes_pair_mode"] = "ordered"
    # Deliberately omit the pair-mode option in the default arm. Older main's
    # default is already ordered and cannot accept the new keyword.
    cfg = make_featurizer_config(CHANNELS, **flags)
    effective_mode = cfg.get("costes_pair_mode", "ordered")
    return cfg, effective_mode


def assert_exact(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("Output shape/dtype changed")
    for predicate in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(predicate(actual), predicate(expected))
    np.testing.assert_array_equal(actual, expected)


def run(path, backend="numpy", revision="", reps=REPS):
    import cp_measure
    from cp_measure import bulk
    from cp_measure.featurizer import featurize

    if reps < 1:
        raise ValueError("reps must be positive")
    cp_measure.set_accelerator(None if backend == "numpy" else backend)
    image, masks = inputs()
    digest = hashlib.sha256(image.tobytes() + masks.tobytes()).hexdigest()
    records, arrays = {}, {}
    for policy in ("defaults", "matched"):
        for mode in ("default", "ordered"):
            key = f"{policy}-{mode}"
            cfg, effective_mode = configuration(policy, mode)
            functions = bulk.get_correlation_measurements(sanitize=False)
            counts = dict.fromkeys(FEATURES, 0)

            def spy(name):
                @wraps(functions[name])
                def measured(**kwargs):
                    # Count actual pair tasks, not registry calls (which batch
                    # pairs), nor individual object-scope threshold fits.
                    counts[name] += len(kwargs["masks"])
                    return functions[name](**kwargs)

                return measured

            # Untimed count validation also warms the entire execution path/JIT.
            with patch.object(
                bulk,
                "get_correlation_measurements",
                return_value={name: spy(name) for name in FEATURES},
            ):
                reference, columns, rows = featurize(
                    image, masks, cfg, image_ids=["ci"]
                )
            expected = {
                "pearson": 20,
                "costes": 10 if effective_mode == "unordered" else 20,
                "manders_fold": 10,
                "rwc": 10,
            }
            if counts != expected:
                raise AssertionError(
                    f"Unexpected pair executions: {counts} != {expected}"
                )
            if len(columns) != len(set(columns)):
                raise AssertionError("Duplicate output columns")
            expected_columns = 100 if effective_mode == "unordered" else 120
            if len(columns) != expected_columns or len(rows) != 4:
                raise AssertionError("Unexpected output coverage")
            timings = []
            for _ in range(reps):
                start = perf_counter()
                values, timed_columns, timed_rows = featurize(
                    image, masks, cfg, image_ids=["ci"]
                )
                timings.append(perf_counter() - start)
                if columns != timed_columns or rows != timed_rows:
                    raise AssertionError("Output metadata changed between calls")
                assert_exact(values, reference)
            arrays[key] = reference
            records[key] = {
                "reps_seconds": timings,
                "pair_executions": counts,
                "columns": columns,
                "rows": rows,
                "config": cfg,
                "effective_mode": effective_mode,
            }
    # featurization must not mutate the reusable benchmark inputs.
    if hashlib.sha256(image.tobytes() + masks.tobytes()).hexdigest() != digest:
        raise AssertionError("Inputs mutated")
    original_image, original_masks = inputs()
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(masks, original_masks)
    dependencies = {name: version(name) for name in ("numpy", "scipy", "scikit-image")}
    dependencies["python"] = platform.python_version()
    if backend == "numba":
        dependencies["numba"] = version("numba")
    np.savez_compressed(Path(path).with_suffix(".npz"), **arrays)
    Path(path).write_text(
        json.dumps(
            {
                "backend": backend,
                "revision": revision,
                "input_sha256": digest,
                "dependencies": dependencies,
                "records": records,
            },
            indent=2,
            default=int,
        )
    )


def reverse_columns():
    return {
        f"Correlation_Costes_{coefficient}__{CHANNELS[a]}__{CHANNELS[b]}"
        for a in range(len(CHANNELS))
        for b in range(a)
        for coefficient in (1, 2)
    }


def compare(base_path, head_path):
    base, head = (json.loads(Path(path).read_text()) for path in (base_path, head_path))
    for key in ("backend", "input_sha256", "dependencies"):
        if base[key] != head[key]:
            raise AssertionError(f"Incompatible benchmark {key}")
    lines = [
        f"### Featurizer colocalization — {head['backend']}",
        "",
        f"Main `{base['revision'][:7]}` vs head `{head['revision'][:7]}`; "
        "five channels, four objects, seeded 64×64 uint8 image; 255-level Costes grid.",
        "Warmed in-memory calls; median [min–max] milliseconds. "
        "Timings informational; pair counts and exact retained outputs are gates. "
        "Matched means colocalization settings, not a CP comparison.",
        "",
        "|Policy / mode|Main ms|Head ms|Main/head|Costes pair tasks main→head|Columns main→head|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    with (
        np.load(Path(base_path).with_suffix(".npz")) as ba,
        np.load(Path(head_path).with_suffix(".npz")) as ha,
    ):
        for key, current in head["records"].items():
            previous = base["records"][key]
            assert_retained(previous, current, ba[key], ha[key])
            # Each environment's default must preserve its ordered control's
            # retained outputs, even after main also adopts the unordered mode.
            ordered = key.split("-")[0] + "-ordered"
            for metadata, data in ((base, ba), (head, ha)):
                assert_retained(
                    metadata["records"][ordered],
                    metadata["records"][key],
                    data[ordered],
                    data[key],
                )
            b, h = (entry["reps_seconds"] for entry in (previous, current))

            def display(times):
                return f"{statistics.median(times) * 1000:.2f} [{min(times) * 1000:.2f}–{max(times) * 1000:.2f}]"

            lines.append(
                f"|{key}|{display(b)}|{display(h)}|{statistics.median(b) / statistics.median(h):.2f}×|"
                f"{previous['pair_executions']['costes']}→{current['pair_executions']['costes']}|"
                f"{len(previous['columns'])}→{len(current['columns'])}|"
            )
    lines += [
        "",
        "Exact retained outputs and NaN/signed-infinity patterns: **PASS**. "
        "Only extra reverse-fit Costes columns may disappear; savings from fewer pairs "
        "are not equal-work algorithm speedups. Three repetitions are not confidence intervals.",
    ]
    return "\n".join(lines)


def assert_retained(previous, current, before, after):
    configs = [dict(record.get("config", {})) for record in (previous, current)]
    for config in configs:
        config.pop("costes_pair_mode", None)
    if configs[0] != configs[1]:
        raise AssertionError("Non-pair configuration changed")
    if previous["rows"] != current["rows"]:
        raise AssertionError("Row metadata changed")
    a, b = previous["columns"], current["columns"]
    expected_removed = (
        reverse_columns()
        if previous["effective_mode"] == "ordered"
        and current["effective_mode"] == "unordered"
        else set()
    )
    if set(a) - set(b) != expected_removed or set(b) - set(a):
        raise AssertionError("Unexpected column coverage change")
    assert_exact(after, before[:, [a.index(column) for column in b]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--backend", choices=("numpy", "numba"), default="numpy")
    run_parser.add_argument("--revision", required=True)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--base", required=True)
    compare_parser.add_argument("--head", required=True)
    compare_parser.add_argument("--md", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.out, args.backend, args.revision)
    else:
        Path(args.md).write_text(compare(args.base, args.head))


if __name__ == "__main__":
    main()
