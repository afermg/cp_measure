import numpy


def get_test_pixels_mask():
    pixels = numpy.random.randint(100, size=64**2).reshape((64, 64))
    mask = numpy.zeros_like(pixels, dtype=bool)
    mask[2:-3, 2:-3] = True
    return pixels, mask


def boolean_mask_to_ijv(mask: numpy.ndarray) -> numpy.ndarray:
    """
    input: 2d boolean array
    output: (n, 3) integer array following (i,j,1)
    """

    # Extract coordinates of object from boolean mask
    i, j = numpy.where(mask)
    n = len(i)
    ijv = numpy.ones((n, 3), dtype=int)
    ijv[:, 0] = i
    ijv[:, 1] = j
    return ijv
