"""Regression tests for ``get_granularity`` (issue #90).

With ``subsample_size=0.25`` and an input axis shorter than 8, ``new_shape``
collapses to 1 along that axis and the back-projection scale factor
``(back_shape[k] - 1) / (new_shape[k] - 1)`` divided by zero. Running without
exception is enough — no assertion needed.
"""

import numpy

from cp_measure.core.measuregranularity import get_granularity


def test_collapsed_axis_2d():
    mask = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        ]
    )
    get_granularity(mask, mask.astype(float))


def test_collapsed_axis_3d():
    # depth=5 -> int(5 * 0.25) = 1, collapses the leading axis
    mask = numpy.zeros((5, 17, 17), dtype=int)
    mask[:, 1:-1, 1:-1] = 1
    get_granularity(mask, mask.astype(float))
