"""Build the (image_size × object_count × seed) fixture matrix once from the pinned ``synth``
generator into ``.npz`` + a sha256 manifest, so ``main`` and the PR head load identical inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy

from cp_measure import synth

# Matrix presets. The default object counts are feasible at the smallest image (512² packs ~180);
# fixed counts across sizes isolate area-scaling (size↑) from per-object cost (count↑).
DEFAULT_MATRIX = {
    "sizes": (512, 1024, 2048),
    "counts": (16, 48, 96, 144),
    "seeds": (0, 1, 2),
}
# CI default: bounded so the full sweep stays well within a hosted runner's memory/time budget
# (the 2048²×144 corner of DEFAULT can OOM/overrun). Use DEFAULT via workflow_dispatch for depth.
CI_MATRIX = {"sizes": (512, 1024), "counts": (16, 64), "seeds": (0, 1)}
SMOKE_MATRIX = {"sizes": (128,), "counts": (4, 8), "seeds": (0,)}

MATRICES = {"default": DEFAULT_MATRIX, "ci": CI_MATRIX, "smoke": SMOKE_MATRIX}

MANIFEST_NAME = "manifest.json"


def spec_key(size: int, n_objects: int, seed: int) -> str:
    return f"s{size}_n{n_objects}_seed{seed}"


def _digest(labels: numpy.ndarray, channels: numpy.ndarray) -> str:
    h = hashlib.sha256()
    h.update(labels.tobytes())
    h.update(channels.tobytes())
    return h.hexdigest()


def build_fixtures(out_dir: str | Path, matrix: dict = DEFAULT_MATRIX) -> dict:
    """Generate every (size, count, seed) fixture into ``out_dir`` and write ``manifest.json``."""
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
                        "sha256": _digest(labels, channels),
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
    """Load ``(labels, channels)`` for one manifest entry, checking its sha256."""
    data = numpy.load(Path(fixtures_dir) / entry["file"])
    labels, channels = data["labels"], data["channels"]
    if _digest(labels, channels) != entry["sha256"]:
        raise ValueError(
            f"fixture {entry['key']} sha256 mismatch — corrupt or wrong generator"
        )
    return labels, channels


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build the benchmark fixture matrix.")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument(
        "--matrix", default="ci", choices=sorted(MATRICES), help="matrix preset"
    )
    a = p.parse_args(argv)
    m = build_fixtures(a.out, MATRICES[a.matrix])
    print(
        f"built {len(m['fixtures'])} fixtures (matrix={a.matrix}, synth v{m['synth_version']}) -> {a.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
