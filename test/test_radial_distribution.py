"""Issue #22: get_radial_distribution must give per-object results independent of
other labels. The default (legacy=False) measures each object on its own crop;
legacy=True keeps the original whole-image behaviour."""

import numpy as np

from cp_measure.core.measureobjectintensitydistribution import get_radial_distribution


def _three_touching():
    """Three abutting rectangles — a layout where the whole-image geometry leaks."""
    labels = np.zeros((48, 48), dtype=np.int32)
    labels[5:40, 5:18] = 1
    labels[5:40, 18:31] = 2
    labels[5:40, 31:44] = 3
    pixels = np.random.default_rng(1).random((48, 48))
    return labels, pixels


def test_radial_per_object_independent_of_neighbours():
    """#22: an object's features are identical with or without other labels present.

    Both sides use the per-object default, so this holds bit-exactly even for
    symmetric objects whose centre is a plateau (the crop is the same array)."""
    labels, pixels = _three_touching()
    new = get_radial_distribution(labels, pixels)
    for lbl in (1, 2, 3):
        alone = np.where(labels == lbl, 1, 0).astype(np.int32)
        iso = get_radial_distribution(alone, pixels)
        for key in new:
            np.testing.assert_allclose(
                new[key][lbl - 1], iso[key][0], equal_nan=True, err_msg=key
            )


def test_radial_legacy_leaks_where_new_does_not():
    """The whole-image (legacy) path is perturbed by neighbours on this layout while
    the per-object default is not — they differ, proving #22 is real and fixed."""
    labels, pixels = _three_touching()
    new = get_radial_distribution(labels, pixels)
    legacy = get_radial_distribution(labels, pixels, legacy=True)
    assert any(not np.allclose(new[k], legacy[k], equal_nan=True) for k in new)


def test_radial_unique_centre_object_new_equals_legacy():
    """For an unambiguous (odd-sized) centre, the per-object crop reproduces the
    whole-image result exactly — confirming the crop introduces no difference of its
    own; the only systematic divergence is the centre tie-break on symmetric objects."""
    labels = np.zeros((41, 41), dtype=np.int32)
    labels[8:29, 8:29] = 1  # 21x21 odd square -> unique centre pixel
    pixels = np.random.default_rng(2).random((41, 41))
    new = get_radial_distribution(labels, pixels)
    legacy = get_radial_distribution(labels, pixels, legacy=True)
    for k in new:
        np.testing.assert_allclose(
            new[k], legacy[k], rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=k
        )
