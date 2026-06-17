"""Tests for the simple synthetic generator."""

import numpy

from cp_measure import synth


def test_count_contiguous_and_dtypes():
    labels, channels = synth.generate(256, 12, n_channels=2, seed=0)
    assert labels.shape == (256, 256) and labels.dtype == numpy.int32
    assert channels.shape == (2, 256, 256) and channels.dtype == numpy.float32
    assert numpy.array_equal(numpy.unique(labels), numpy.arange(0, 13))  # bg + 1..12


def test_determinism():
    a = synth.generate(128, 9, seed=1)
    b = synth.generate(128, 9, seed=1)
    assert numpy.array_equal(a[0], b[0]) and numpy.array_equal(a[1], b[1])
    assert not numpy.array_equal(
        synth.generate(128, 9, seed=2)[1], a[1]
    )  # channels re-randomise
