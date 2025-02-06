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


def get_transposed_mask(size: int = 789):
    masks = get_sample_label_masks(size=size)
    return [masks, masks.T]


@pytest.mark.parametrize("masks", (get_sample_label_masks(),))
@pytest.mark.parametrize(
    "distance_method",
    ("Adjacent", "Expand until adjacent", "Within a specified distance"),
)
def test_neighbors(masks: numpy.ndarray, distance_method: str):
    return measureobjectneighbors(masks, masks, distance_method=distance_method)


@pytest.mark.parametrize("masks1", get_transposed_mask())
@pytest.mark.parametrize("masks2", get_transposed_mask())
@pytest.mark.parametrize("decimation_method", ("K means", "Skeleton"))
@pytest.mark.parametrize("wants_emd", (True, False))
def test_overlap(
    masks1: numpy.ndarray,
    masks2: numpy.ndarray,
    decimation_method: str,
    wants_emd: bool,
):
    return measureobjectoverlap(
        masks1, masks2, decimation_method=decimation_method, wants_emd=wants_emd
    )
