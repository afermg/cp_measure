"""Regression checks for the synthetic benchmark generator ``cp_measure.synth``."""

import numpy
import pytest
import scipy.ndimage
from skimage.measure import regionprops_table

from cp_measure import synth

CONFIG = (512, 80)  # enough objects for stats, fast to generate


def test_generate_invariants():
    size, n = CONFIG
    labels, channels = synth.generate(size, n, n_channels=2, seed=0)
    # shape / dtype / contiguous 1..n labels at the requested count (cp_measure's contract)
    assert labels.shape == (size, size) and labels.dtype == numpy.int32
    assert channels.shape == (2, size, size) and channels.dtype == numpy.float32
    assert numpy.array_equal(numpy.unique(labels)[1:], numpy.arange(1, n + 1))
    # no degenerate (~1px) objects; some shape and size variety
    areas = numpy.bincount(labels.ravel())[1:]
    assert areas.min() >= 20
    rp = regionprops_table(labels, properties=("eccentricity", "area"))
    assert rp["eccentricity"].max() - rp["eccentricity"].min() > 0.2
    assert rp["area"].max() / rp["area"].min() > 2
    # real signal: cells brighter than background, texture above noise, coloc in a controlled band
    fg = labels > 0
    assert channels[0][fg].mean() > channels[0][~fg].mean()
    stds = scipy.ndimage.standard_deviation(channels[0], labels, numpy.arange(1, n + 1))
    assert numpy.median(stds) > 5 * synth._NOISE_LEVEL
    r = numpy.corrcoef(channels[0][fg], channels[1][fg])[0, 1]
    assert 0.2 < r < 0.9


def test_determinism():
    a = synth.generate(256, 20, seed=1)
    b = synth.generate(256, 20, seed=1)
    assert numpy.array_equal(a[0], b[0]) and numpy.array_equal(a[1], b[1])
    assert not numpy.array_equal(synth.generate(256, 20, seed=2)[0], a[0])


def test_edges():
    labels, channels = synth.generate(
        128, 0, n_channels=1, seed=0
    )  # empty + single channel
    assert labels.max() == 0 and channels.shape == (1, 128, 128)
    with pytest.raises(ValueError, match="cannot fit"):
        synth.generate(256, 5000, seed=0)
