"""Tests for central non-contiguous label sanitation (cp_measure._sanitize)."""

import numpy as np
import pytest

from cp_measure._detect import HAS_NUMBA
from cp_measure._sanitize import sanitize_masks
from cp_measure.bulk import (
    get_core_measurements,
    get_core_measurements_3d,
    get_correlation_measurements,
    get_multimask_measurements,
)
from cp_measure.core.measurecolocalization import get_correlation_overlap
from cp_measure.featurizer import featurize, make_featurizer_config


def _three_objects(labels):
    """3 squares in a 64x64 frame, labelled with the given (l1, l2, l3)."""
    m = np.zeros((64, 64), dtype=np.int32)
    m[5:19, 5:19] = labels[0]
    m[5:19, 30:44] = labels[1]
    m[35:49, 5:19] = labels[2]
    return m


# --- sanitize_masks policy -------------------------------------------------

_M3D = np.zeros((4, 16, 16), dtype=np.int32)
_M3D[0, 0:4, 0:4] = 3
_M3D[3, 10:14, 10:14] = 9
_MPIX = np.zeros((8, 8), dtype=np.int32)
_MPIX[3, 3] = 42


@pytest.mark.parametrize(
    "mask, clean_labels, ids",
    [
        (_three_objects((1, 2, 3)), [1, 2, 3], [1, 2, 3]),  # contiguous
        (_three_objects((1, 17, 5)), [1, 2, 3], [1, 5, 17]),  # gapped
        (_M3D, [1, 2], [3, 9]),  # 3D
        (_MPIX, [1], [42]),  # single pixel
        (np.zeros((8, 8), np.int32), [], []),  # all background
    ],
)
def test_sanitize_relabels(mask, clean_labels, ids):
    clean, got_ids = sanitize_masks(mask)
    assert sorted(set(np.unique(clean)) - {0}) == clean_labels
    assert got_ids.tolist() == ids


@pytest.mark.parametrize(
    "mask, same_object",
    [
        (_three_objects((1, 2, 3)), True),  # contiguous -> returned as-is
        (_three_objects((1, 17, 5)), False),  # relabelled -> fresh copy
    ],
)
def test_no_mutation_and_copy_policy(mask, same_object):
    before = mask.copy()
    clean, _ids = sanitize_masks(mask)
    assert (clean is mask) == same_object
    assert np.array_equal(mask, before)


def test_bool_mask_is_single_object():
    m = np.zeros((8, 8), dtype=bool)
    m[2:5, 2:5] = True
    _clean, ids = sanitize_masks(m)
    assert ids.tolist() == [1]


@pytest.mark.parametrize("bad", [np.array([[0, -1, 2]]), np.array([[0.0, 1.0, 2.0]])])
def test_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        sanitize_masks(bad)


# --- decorator coverage ----------------------------------------------------


def _is_sanitized(fn):
    # unwrap functools.partial (legacy binding) before checking the flag
    return bool(
        getattr(fn, "_sanitized", False)
        or getattr(getattr(fn, "func", None), "_sanitized", False)
    )


def test_all_single_mask_registry_funcs_sanitized():
    for reg in (
        get_core_measurements(),
        get_core_measurements(legacy=True),  # partial-wrapped intensity
        get_core_measurements_3d(),
        get_correlation_measurements(),
    ):
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


def test_multimask_funcs_left_unsanitized():
    # two-mask functions are out of scope and must not be wrapped
    assert all(not _is_sanitized(fn) for fn in get_multimask_measurements().values())


# --- per-function: gapped result == contiguous baseline --------------------
# CONTIG and GAPPED share geometry and rank order, so a correct relabel makes
# every feature's output identical. Labels are non-monotonic vs spatial order,
# so this also exercises the rank remap, not just value substitution.
CONTIG = _three_objects((1, 3, 2))
GAPPED = _three_objects((1, 17, 5))
_RNG = np.random.default_rng(0)
PIX1 = _RNG.random((64, 64))
PIX2 = _RNG.random((64, 64))


def _assert_close_dicts(a, b):
    assert list(a.keys()) == list(b.keys())
    for k in a:
        np.testing.assert_allclose(
            np.asarray(a[k], float),
            np.asarray(b[k], float),
            equal_nan=True,
            err_msg=f"mismatch in feature {k!r}",
        )


@pytest.mark.parametrize("name", list(get_core_measurements()))
def test_core_func_gapped_matches_contiguous(name):
    func = get_core_measurements()[name]
    _assert_close_dicts(func(GAPPED, PIX1), func(CONTIG, PIX1))


@pytest.mark.parametrize(
    "func",
    list(get_correlation_measurements().values()) + [get_correlation_overlap],
    ids=lambda f: f.__name__,
)
def test_correlation_func_gapped_matches_contiguous(func):
    _assert_close_dicts(
        func(pixels_1=PIX1, pixels_2=PIX2, masks=GAPPED),
        func(pixels_1=PIX1, pixels_2=PIX2, masks=CONTIG),
    )


# --- featurizer end-to-end -------------------------------------------------


def test_featurizer_gapped_reports_original_ids():
    image = np.stack([PIX1, PIX2], axis=0)
    config = make_featurizer_config(["DNA", "ER"])
    data_g, cols_g, rows_g = featurize(image, GAPPED[np.newaxis], config)
    data_c, cols_c, rows_c = featurize(image, CONTIG[np.newaxis], config)
    assert cols_g == cols_c
    assert rows_g == [(None, "object", 1), (None, "object", 5), (None, "object", 17)]
    assert rows_c == [(None, "object", 1), (None, "object", 2), (None, "object", 3)]
    np.testing.assert_allclose(data_g, data_c, equal_nan=True)


def test_featurizer_does_not_mutate_input():
    image = np.stack([PIX1, PIX2], axis=0)
    config = make_featurizer_config(["DNA", "ER"])
    masks = GAPPED[np.newaxis].copy()
    before = masks.copy()
    featurize(image, masks, config)
    assert np.array_equal(masks, before)
