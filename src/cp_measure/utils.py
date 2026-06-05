""" "
Utilities reused in multiple measurements.
"""

import numpy


def _ensure_np_array(value):
    """Convert a result from scipy.ndimage to a numpy array

    scipy.ndimage has the annoying habit of returning a single, bare
    value instead of an array if the indexes passed in are of length 1.
    For instance:
    scind.maximum(image, labels, [1]) returns a float
    but
    scind.maximum(image, labels, [1,2]) returns a list
    """
    return numpy.array([value]) if numpy.isscalar(value) else numpy.array(value)


def _ensure_np_scalar(value):
    return value if numpy.isscalar(value) else numpy.array(value).squeeze()


def get_test_pixels_mask():
    pixels = numpy.random.randint(100, size=64**2).reshape((64, 64))
    mask = numpy.zeros_like(pixels, dtype=bool)
    mask[2:-3, 2:-3] = True
    return pixels, mask


def masks_to_ijv(masks: numpy.ndarray) -> numpy.ndarray:
    """
    input: 2d integer label array
    output: (n, 3) integer array of rows (i, j, label) sorted by label
    """
    i, j = numpy.nonzero(masks)
    v = masks[i, j]
    order = numpy.argsort(v, kind="stable")
    return numpy.column_stack((i[order], j[order], v[order])).astype(int, copy=False)


def labels_to_binmasks(masks: numpy.ndarray) -> numpy.ndarray:
    """
    Convert a label matrix to a boolean masks.

    Returns a list of binary masks.
    """
    labels = numpy.unique(masks)
    labels = labels[labels > 0]
    return masks == labels.reshape((-1,) + (1,) * masks.ndim)
