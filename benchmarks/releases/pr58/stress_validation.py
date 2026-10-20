"""Extra seeded numerical stress probe; run in the rebased project's environment."""

import json
import warnings

import numpy as np

from cp_measure._sanitize import sanitize_masks
from cp_measure.core.measureobjectsizeshape import get_zernike as shape_numpy
from cp_measure.core.numba.measureobjectsizeshape import get_zernike as shape_numba
from cp_measure.core.measureobjectintensitydistribution import (
    get_radial_zernikes as radial_numpy,
)
from cp_measure.core.numba.measureobjectintensitydistribution import (
    get_radial_zernikes as radial_numba,
)


def main():
    count, maximum = 0, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(80):
            rng = np.random.default_rng(seed)
            h, w = rng.integers(16, 130, 2)
            y, x = np.mgrid[:h, :w]
            mask = np.zeros((h, w), np.int32)
            for k in range(1, 5):
                cy, cx = rng.uniform(0, h), rng.uniform(0, w)
                a, b = rng.uniform(1, 15, 2)
                mask[((y - cy) / a) ** 2 + ((x - cx) / b) ** 2 <= 1] = k
            mask, _ = sanitize_masks(mask)
            for pixels in [rng.random((h, w)) - 0.1, np.ones((h, w))]:
                for ref, got in [
                    (shape_numpy, shape_numba),
                    (radial_numpy, radial_numba),
                ]:
                    aa, bb = ref(mask, pixels), got(mask, pixels)
                    assert list(aa) == list(bb)
                    for key in aa:
                        np.testing.assert_allclose(
                            aa[key],
                            bb[key],
                            rtol=1e-6,
                            atol=1e-8,
                            equal_nan=True,
                            err_msg=f"{seed}/{key}",
                        )
                        finite = np.isfinite(aa[key]) & np.isfinite(bb[key])
                        if finite.any():
                            maximum = max(
                                maximum,
                                float(
                                    np.max(np.abs(aa[key][finite] - bb[key][finite]))
                                ),
                            )
                    count += 1
    print(
        json.dumps(
            dict(
                seed_range=[0, 79],
                cases=count,
                max_abs_difference=maximum,
                passed=True,
                description="Irregular elliptical masks, overlapping/disconnected labels, signed random and constant intensities, two backends",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
