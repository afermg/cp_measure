""" "
Utilities reused in multiple measurements.
"""

import numpy


def get_test_pixels_mask():
    pixels = numpy.random.randint(100, size=64**2).reshape((64, 64))
    mask = numpy.zeros_like(pixels, dtype=bool)
    mask[2:-3, 2:-3] = True
    return pixels, mask


def masks_to_ijv(masks: numpy.ndarray) -> numpy.ndarray:
    """
    input: 2d boolean array
    output: (n, 3) integer array following (i,j,1)
    """

    # Extract coordinates of object from boolean mask
    masks_ijv = numpy.empty((0, 3), dtype=int)
    for label in range(masks.max()):
        i, j = numpy.where(masks == label + 1)
        n = len(i)
        ijv = numpy.empty((n, 3), dtype=int)
        ijv[:, 0] = i
        ijv[:, 1] = j
        ijv[:, 2] = label + 1
        masks_ijv = numpy.concatenate((masks_ijv, ijv))

    return masks_ijv


def labels_to_binmasks(masks: numpy.ndarray) -> numpy.ndarray:
    """
    Convert a label matrix to a boolean masks.

    Returns a list of binary masks.
    """
    labels = numpy.unique(masks)
    labels = sorted(labels[labels > 0])
    return [(masks == i) for i in labels]
