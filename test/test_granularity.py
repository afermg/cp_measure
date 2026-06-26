"""Regression tests for ``get_granularity`` (issue #90).

With ``subsample_size=0.25``, axes of length 4-7 collapse ``new_shape[k]`` to 1
(div-by-zero in the back-projection), and axes <=3 collapse it to 0 (empty-array
crash in morphology). Running without exception is enough — no assertion needed.
"""

import numpy

from cp_measure.core.measuregranularity import get_granularity


def _ring_mask(shape):
    mask = numpy.zeros(shape, dtype=int)
    if mask.ndim == 2:
        mask[:, 1:-1] = 1
    else:
        mask[:, :, 1:-1] = 1
    return mask


# new_shape[k] == 1 (axis length 4-7)
def test_collapsed_to_one_2d():
    get_granularity(_ring_mask((7, 17)), numpy.zeros((7, 17)))


def test_collapsed_to_one_3d():
    get_granularity(_ring_mask((5, 17, 17)), numpy.zeros((5, 17, 17)))


# new_shape[k] == 0 before clamp (axis length 1-3)
def test_collapsed_to_zero_2d():
    get_granularity(_ring_mask((3, 17)), numpy.zeros((3, 17)))


def test_collapsed_to_zero_3d():
    get_granularity(_ring_mask((2, 17, 17)), numpy.zeros((2, 17, 17)))
