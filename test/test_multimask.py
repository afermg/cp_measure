""" "
Test functions that operate multiple pixel arrays (e.g., correlations).
"""

import numpy
import pytest
from cp_measure.multimask.measureobjectneighbors import measureobjectneighbors
from cp_measure.multimask.measureobjectoverlap import measureobjectoverlap


def get_sample_label_masks(size: int = 789):
    masks = numpy.zeros((size, size), dtype=int)
    masks[:100, 100] = 1
    masks[300:400, 300:400] = 2
    masks[-100:, -100:] = 3
    return masks


@pytest.mark.parametrize("masks", (get_sample_label_masks(),))
@pytest.mark.parametrize(
    "distance_method",
    ("Adjacent", "Expand until adjacent", "Within a specified distance"),
)
def test_neighbors(masks: numpy.ndarray, distance_method: str):
    return measureobjectneighbors(masks, masks, distance_method=distance_method)


@pytest.mark.parametrize("masks", (get_sample_label_masks(),))
@pytest.mark.parametrize("decimation_method", ("K means", "Skeleton"))
def test_overlap(masks: numpy.ndarray, decimation_method: str):
    return measureobjectoverlap(masks, masks, decimation_method=decimation_method)
