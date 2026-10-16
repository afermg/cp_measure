"""Reproduce the pre-batch featurizer fixture using the pinned source checkout.

Run from this PR's checkout with the OLD checkout on PYTHONPATH, for example:

    git worktree add --detach /tmp/cp-measure-reference 1f2ef4d
    PYTHONPATH=/tmp/cp-measure-reference/src python test/make_featurizer_reference.py

The generator refuses a different or locally modified source tree. Tests only
load the saved JSON; they never need Git or the old checkout.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np

REFERENCE_COMMIT = "1f2ef4dc1fa0bfd81c2b3203b9e31af8d4e2e8c0"


def make_reference_case(is_3d, legacy=False):
    """Two distinct images, two channels, and non-contiguous IDs in two mask types."""
    from cp_measure.featurizer import make_featurizer_config

    spatial = (6, 32, 32) if is_3d else (32, 32)
    image = np.random.default_rng(106).random((2, 2, *spatial))
    masks_2d = np.zeros((2, 2, 32, 32), dtype=np.int32)
    masks_2d[0, 0, 3:10, 4:13] = 3
    masks_2d[0, 0, 18:26, 20:29] = 17
    masks_2d[0, 1, 2:15, 3:18] = 5
    # The second image has no nuclei and three cells, with different IDs.
    masks_2d[1, 1, 2:9, 3:13] = 9
    masks_2d[1, 1, 12:21, 17:28] = 42
    masks_2d[1, 1, 24:30, 2:10] = 100
    if is_3d:
        masks = np.zeros((2, 2, *spatial), dtype=np.int32)
        masks[:, :, 1:5] = masks_2d[:, :, None]
    else:
        masks = masks_2d
    config = make_featurizer_config(
        ["DNA", "ER"],
        objects=["nuclei", "cells"],
        legacy=legacy,
        texture=False,
        granularity=False,
        radial_distribution=False,
        radial_zernikes=False,
        zernike=False,
        feret=False,
        correlation_costes=False,
        correlation_manders_fold=False,
        correlation_rwc=False,
    )
    # Intensity, size/shape, and directional Pearson correlations remain enabled.
    return image, masks, config


def main():
    import cp_measure
    from cp_measure.featurizer import featurize

    source_root = Path(cp_measure.__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--", "src"], text=True
    ).strip()
    if commit != REFERENCE_COMMIT or changes:
        raise RuntimeError("Set PYTHONPATH to the clean, pinned pre-refactor checkout")
    source = source_root / "src/cp_measure/featurizer.py"
    result = {
        "source_commit": commit,
        "featurizer_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-image", "centrosome")
        },
        "cases": {},
    }
    for is_3d in (False, True):
        for legacy in (False, True):
            image, masks, config = make_reference_case(is_3d, legacy)
            outputs = [
                featurize(image[b], masks[b], config, image_id=image_id)
                for b, image_id in enumerate(["A01", "A02"])
            ]
            assert outputs[0][1] == outputs[1][1]
            data = np.vstack([part[0] for part in outputs])
            # JSON null records undefined measurements without non-standard NaN tokens.
            values = [[None if np.isnan(v) else float(v) for v in row] for row in data]
            key = f"{'3d' if is_3d else '2d'}-{'legacy' if legacy else 'modern'}"
            result["cases"][key] = {
                "columns": outputs[0][1],
                "rows": [
                    [str(i), str(o), int(label)]
                    for _, _, rows in outputs
                    for i, o, label in rows
                ],
                "data": values,
            }
    target = Path(__file__).parent / "data/featurizer_pre_batch.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(target)


if __name__ == "__main__":
    main()
