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
    final_ijv = np.empty((0,3), dtype=int)
    for label in range(masks.max()):
        i, j = numpy.where(mask==label+1)
        n = len(i)
        ijv = numpy.ones((n, 3), dtype=int)
        ijv[:, 0] = i
        ijv[:, 1] = j
        ijv[:,2] = label

    
    return final_ijv
