"""Tests for central non-contiguous label sanitation (cp_measure._sanitize)."""

import numpy as np
import pytest

from cp_measure._detect import HAS_NUMBA
from cp_measure._sanitize import _is_contiguous, sanitize_labels, sanitize_masks
from cp_measure.bulk import (
    get_core_measurements,
    get_core_measurements_3d,
    get_correlation_measurements,
)
from cp_measure.featurizer import featurize, make_featurizer_config


# --------------------------------------------------------------------------
# sanitize_masks policy
# --------------------------------------------------------------------------


def _three_objects(labels):
    """3 squares in a 64x64 frame, labelled with the given (l1, l2, l3)."""
    m = np.zeros((64, 64), dtype=np.int32)
    m[5:19, 5:19] = labels[0]
    m[5:19, 30:44] = labels[1]
    m[35:49, 5:19] = labels[2]
    return m


def test_gapped_labels_relabelled_to_1_n():
    gapped = _three_objects((1, 5, 17))
    clean, ids = sanitize_masks(gapped)
    assert sorted(set(np.unique(clean)) - {0}) == [1, 2, 3]
    assert ids.tolist() == [1, 5, 17]


def test_input_not_mutated():
    gapped = _three_objects((1, 5, 17))
    before = gapped.copy()
    clean, _ids = sanitize_masks(gapped)
    assert np.array_equal(gapped, before)
    assert clean is not gapped  # relabel path returns a fresh copy


def test_contiguous_fast_path_returns_input_unchanged():
    contig = _three_objects((1, 2, 3))
    clean, ids = sanitize_masks(contig)
    assert clean is contig  # no copy when already 1..N
    assert ids.tolist() == [1, 2, 3]


def test_all_background():
    clean, ids = sanitize_masks(np.zeros((8, 8), dtype=np.int32))
    assert ids.size == 0


def test_3d_gapped():
    m = np.zeros((4, 16, 16), dtype=np.int32)
    m[0, 0:4, 0:4] = 3
    m[3, 10:14, 10:14] = 9
    clean, ids = sanitize_masks(m)
    assert sorted(set(np.unique(clean)) - {0}) == [1, 2]
    assert ids.tolist() == [3, 9]


def test_single_pixel():
    m = np.zeros((8, 8), dtype=np.int32)
    m[3, 3] = 42
    clean, ids = sanitize_masks(m)
    assert ids.tolist() == [42]
    assert clean[3, 3] == 1


def test_is_contiguous():
    assert _is_contiguous(_three_objects((1, 2, 3)))
    assert not _is_contiguous(_three_objects((1, 5, 17)))
    assert _is_contiguous(np.zeros((4, 4), dtype=np.int32))


@pytest.mark.parametrize(
    "bad",
    [np.array([[0, -1, 2]]), np.array([[0.0, 1.0, 2.0]])],
)
def test_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        sanitize_masks(bad)


# --------------------------------------------------------------------------
# decorator coverage (every registry function is sanitized)
# --------------------------------------------------------------------------


def _is_sanitized(fn):
    # unwrap functools.partial (legacy binding) before checking the flag
    return bool(
        getattr(fn, "_sanitized", False)
        or getattr(getattr(fn, "func", None), "_sanitized", False)
    )


def test_all_single_mask_registry_funcs_sanitized():
    registries = [
        get_core_measurements(),
        get_core_measurements(legacy=True),  # partial-wrapped intensity
        get_core_measurements_3d(),
        get_correlation_measurements(),
    ]
    for reg in registries:
        for name, fn in reg.items():
            assert _is_sanitized(fn), f"{name} is not @sanitize_labels-wrapped"


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_numba_registry_funcs_sanitized():
    import cp_measure

    cp_measure.set_accelerator("numba")
    try:
        for name, fn in get_core_measurements().items():
            assert _is_sanitized(fn), f"numba {name} is not sanitized"
    finally:
        cp_measure.set_accelerator(None)


def test_decorator_leaves_unknown_signature_untouched():
    def two_mask(masks1, masks2):  # multimask-style: no recognised label arg
        return masks1

    assert sanitize_labels(two_mask) is two_mask


# --------------------------------------------------------------------------
# per-function: gapped result == contiguous baseline (direct-call coverage)
# --------------------------------------------------------------------------

CONTIG = _three_objects((1, 2, 3))
GAPPED = _three_objects((1, 5, 17))  # same geometry, ascending -> same ranks
_RNG = np.random.default_rng(0)
PIX1 = _RNG.random((64, 64))
PIX2 = _RNG.random((64, 64))


def _allclose_dicts(a, b):
    assert list(a.keys()) == list(b.keys())
    for k in a:
        np.testing.assert_allclose(
            np.asarray(a[k], dtype=float),
            np.asarray(b[k], dtype=float),
            equal_nan=True,
            err_msg=f"mismatch in feature {k!r}",
        )


@pytest.mark.parametrize("name", list(get_core_measurements().keys()))
def test_core_func_gapped_matches_contiguous(name):
    func = get_core_measurements()[name]
    _allclose_dicts(func(GAPPED, PIX1), func(CONTIG, PIX1))


@pytest.mark.parametrize("name", list(get_correlation_measurements().keys()))
def test_correlation_func_gapped_matches_contiguous(name):
    func = get_correlation_measurements()[name]
    _allclose_dicts(
        func(pixels_1=PIX1, pixels_2=PIX2, masks=GAPPED),
        func(pixels_1=PIX1, pixels_2=PIX2, masks=CONTIG),
    )


# --------------------------------------------------------------------------
# featurizer end-to-end
# --------------------------------------------------------------------------


def test_featurizer_gapped_reports_original_ids():
    image = np.stack([PIX1, PIX2], axis=0)
    config = make_featurizer_config(["DNA", "ER"])

    gapped_masks = GAPPED[np.newaxis]
    contig_masks = CONTIG[np.newaxis]

    data_g, cols_g, rows_g = featurize(image, gapped_masks, config)
    data_c, cols_c, rows_c = featurize(image, contig_masks, config)

    assert cols_g == cols_c
    # rows carry the ORIGINAL ids, not 1..N
    assert rows_g == [(None, "object", 1), (None, "object", 5), (None, "object", 17)]
    assert rows_c == [(None, "object", 1), (None, "object", 2), (None, "object", 3)]
    # and the actual feature values are identical (same geometry)
    np.testing.assert_allclose(data_g, data_c, equal_nan=True)


def test_featurizer_does_not_mutate_input():
    image = np.stack([PIX1, PIX2], axis=0)
    config = make_featurizer_config(["DNA", "ER"])
    masks = GAPPED[np.newaxis].copy()
    before = masks.copy()
    featurize(image, masks, config)
    assert np.array_equal(masks, before)
