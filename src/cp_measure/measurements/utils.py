import numpy


def get_test_pixels_mask():
    pixels = numpy.random.randint(100, size=64**2).reshape((64, 64))
    mask = numpy.ones_like(pixels, dtype=bool)
    return pixels, mask
