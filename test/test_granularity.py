"""Regression test for ``get_granularity`` (issue #90).

When subsampling would shrink any axis below 2 samples, the bilinear
back-projection has no well-defined geometry; ``get_granularity`` short-circuits
and returns NaN per object instead of crashing.
"""

import numpy

from cp_measure.core.measuregranularity import get_granularity


def test_returns_nan_when_subsampling_collapses_axis():
    mask = numpy.zeros((7, 17), dtype=int)
    mask[:, 1:-1] = 1
    out = get_granularity(mask, numpy.zeros((7, 17)))
    for v in out.values():
        assert numpy.isnan(v).all()
