"""Acceptance tests for the synthetic benchmark generator ``cp_measure.synth``.

These replace the (non-reproducible) "eyeball the rendered examples" gate from the design plan
with programmatic asserts run at the **matrix corners** — the smallest image at the highest
object count and the largest image at the lowest — where the generator is most likely to
degenerate. They lock the guarantees the benchmark harness relies on: determinism, contiguous
labels with the requested object count, no degenerate (≈1px) objects, real shape/texture/intensity
signal, and a controlled (non-trivial, sub-unity) cross-channel correlation for colocalisation.
"""

import numpy
import pytest
import scipy.ndimage
from skimage.measure import regionprops_table

from cp_measure import synth

# Matrix corners: (image_size, n_objects). MAX_COUNT is chosen feasible at the smallest image
# (512² packs ~180 cells; 120 leaves headroom so placement never hits the attempt budget).
MIN_SIZE_MAX_COUNT = (512, 120)
MAX_SIZE_MIN_COUNT = (2048, 16)
CORNERS = [MIN_SIZE_MAX_COUNT, MAX_SIZE_MIN_COUNT]
MID = (1024, 64)

# Degenerate-object floor: cp_measure shape features return zeros silently on ~1px objects, so the
# generator must never emit one. _MIN_RADIUS=3.5 ⇒ areas comfortably above this.
MIN_OBJECT_AREA = 20


def _foreground_areas(labels):
    counts = numpy.bincount(labels.ravel())
    return counts[1:][counts[1:] > 0]


def _per_object_std(img, labels):
    idx = numpy.arange(1, int(labels.max()) + 1)
    return numpy.asarray(scipy.ndimage.standard_deviation(img, labels, idx))


@pytest.mark.parametrize("size,n", CORNERS)
def test_deterministic(size, n):
    labels_a, ch_a = synth.generate(size, n, n_channels=2, seed=7)
    labels_b, ch_b = synth.generate(size, n, n_channels=2, seed=7)
    assert numpy.array_equal(labels_a, labels_b)
    assert numpy.array_equal(ch_a, ch_b)


def test_seed_changes_output():
    labels_0, ch_0 = synth.generate(*MID, n_channels=2, seed=0)
    labels_1, ch_1 = synth.generate(*MID, n_channels=2, seed=1)
    assert not numpy.array_equal(labels_0, labels_1)
    assert not numpy.array_equal(ch_0, ch_1)


@pytest.mark.parametrize("size,n", CORNERS)
def test_shapes_and_dtypes(size, n):
    labels, channels = synth.generate(size, n, n_channels=2, seed=0)
    assert labels.shape == (size, size)
    assert labels.dtype == numpy.int32
    assert channels.shape == (2, size, size)
    assert channels.dtype == numpy.float32
    assert channels.min() >= 0.0


@pytest.mark.parametrize("size,n", CORNERS)
def test_contiguous_labels_and_exact_count(size, n):
    labels, _ = synth.generate(size, n, n_channels=2, seed=3)
    present = numpy.unique(labels)
    present = present[present > 0]
    assert present.size == n, "realized object count must equal the request"
    # cp_measure's contract: labels are the contiguous range 1..n.
    assert numpy.array_equal(present, numpy.arange(1, n + 1))


@pytest.mark.parametrize("size,n", CORNERS)
def test_no_degenerate_objects(size, n):
    labels, _ = synth.generate(size, n, n_channels=2, seed=5)
    areas = _foreground_areas(labels)
    assert areas.min() >= MIN_OBJECT_AREA, (
        "no object may collapse near the degenerate floor"
    )
    # Every object must have a measurable extent (no zero-perimeter / single-pixel cells).
    rp = regionprops_table(
        labels, properties=("area", "perimeter", "axis_minor_length")
    )
    assert numpy.all(rp["perimeter"] > 0)
    assert numpy.all(rp["axis_minor_length"] > 0)


def test_shape_variety_is_organic():
    # Multi-object config: cells must vary in elongation and be non-convex (organic), not disks.
    labels, _ = synth.generate(*MID, n_channels=2, seed=0)
    rp = regionprops_table(labels, properties=("eccentricity", "solidity", "area"))
    assert rp["eccentricity"].max() - rp["eccentricity"].min() > 0.2, (
        "shapes too uniform"
    )
    assert rp["solidity"].min() < 0.99, "boundaries are convex disks, not organic cells"
    areas = rp["area"]
    assert areas.max() / areas.min() > 2.0, "no size variety (big vs tiny cells)"


@pytest.mark.parametrize("size,n", CORNERS)
def test_intensity_is_object_correlated(size, n):
    # The shared envelope must make cells brighter than background (intensity features need signal).
    labels, channels = synth.generate(size, n, n_channels=2, seed=2)
    fg = labels > 0
    assert channels[0][fg].mean() > channels[0][~fg].mean()


@pytest.mark.parametrize("size,n", CORNERS)
def test_texture_signal_within_objects(size, n):
    # Haralick/texture features need intra-object intensity structure: per-object std must be > 0.
    labels, channels = synth.generate(size, n, n_channels=2, seed=4)
    stds = _per_object_std(channels[0], labels)
    assert numpy.all(stds > 0), "objects are flat — no texture signal"


@pytest.mark.parametrize("size,n", CORNERS)
def test_channel_correlation_is_controlled(size, n):
    # Colocalisation inputs must be correlated (shared structure) but not degenerate or ~identical.
    labels, channels = synth.generate(size, n, n_channels=2, seed=1)
    fg = labels > 0
    r = numpy.corrcoef(channels[0][fg], channels[1][fg])[0, 1]
    assert 0.2 < r < 0.9, f"channel correlation {r:.2f} outside the controlled band"


def test_channel_correlation_band_seed_averaged():
    # Tighter target band on the seed-averaged correlation at a representative config (less noisy
    # than any single sparse corner).
    rs = []
    for seed in range(8):
        labels, channels = synth.generate(*MID, n_channels=2, seed=seed)
        fg = labels > 0
        rs.append(numpy.corrcoef(channels[0][fg], channels[1][fg])[0, 1])
    assert 0.4 <= numpy.mean(rs) <= 0.7


def test_single_channel_for_core_features():
    labels, channels = synth.generate(*MID, n_channels=1, seed=0)
    assert channels.shape == (1, *labels.shape)


def test_zero_objects_is_empty():
    labels, channels = synth.generate(256, 0, n_channels=2, seed=0)
    assert labels.max() == 0
    assert channels.shape == (2, 256, 256)


def test_infeasible_count_raises():
    with pytest.raises(ValueError, match="cannot fit"):
        synth.generate(256, 5000, n_channels=2, seed=0)
