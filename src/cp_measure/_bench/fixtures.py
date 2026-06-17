"""Build the (image_size × object_count × seed) fixture matrix once into ``.npz`` + a manifest,
so the ``main`` and PR-head runs load identical inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy

from cp_measure import synth

DEFAULT_MATRIX = {"sizes": (512, 1024), "counts": (16, 64), "seeds": (0, 1)}
CI_MATRIX = {"sizes": (512,), "counts": (16, 64), "seeds": (0, 1)}
SMOKE_MATRIX = {"sizes": (128,), "counts": (4, 8), "seeds": (0,)}
MATRICES = {"default": DEFAULT_MATRIX, "ci": CI_MATRIX, "smoke": SMOKE_MATRIX}

MANIFEST_NAME = "manifest.json"


def spec_key(size: int, n_objects: int, seed: int) -> str:
    return f"s{size}_n{n_objects}_seed{seed}"


def build_fixtures(out_dir: str | Path, matrix: dict = DEFAULT_MATRIX) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for size in matrix["sizes"]:
        for n_objects in matrix["counts"]:
            for seed in matrix["seeds"]:
                labels, channels = synth.generate(
                    size, n_objects, n_channels=2, seed=seed
                )
                key = spec_key(size, n_objects, seed)
                numpy.savez(out / f"{key}.npz", labels=labels, channels=channels)
                entries.append(
                    {
                        "key": key,
                        "size": size,
                        "n_objects": n_objects,
                        "seed": seed,
                        "file": f"{key}.npz",
                    }
                )
    manifest = {
        "synth_version": synth.__version__,
        "matrix": {k: list(v) for k, v in matrix.items()},
        "fixtures": entries,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(fixtures_dir: str | Path) -> dict:
    return json.loads((Path(fixtures_dir) / MANIFEST_NAME).read_text())


def load_fixture(fixtures_dir: str | Path, entry: dict):
    data = numpy.load(Path(fixtures_dir) / entry["file"])
    return data["labels"], data["channels"]


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build the benchmark fixture matrix.")
    p.add_argument("--out", required=True)
    p.add_argument("--matrix", default="ci", choices=sorted(MATRICES))
    a = p.parse_args(argv)
    m = build_fixtures(a.out, MATRICES[a.matrix])
    print(f"built {len(m['fixtures'])} fixtures (matrix={a.matrix}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
