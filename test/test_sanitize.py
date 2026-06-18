"""Tests for central non-contiguous label sanitation (cp_measure._sanitize)."""

import numpy as np
import pytest

from cp_measure._sanitize import sanitize_masks
from cp_measure.bulk import (
    get_core_measurements,
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


_M3D = np.zeros((4, 16, 16), dtype=np.int32)
_M3D[0, :4, :4] = 3
_M3D[3, 10:14, 10:14] = 9


@pytest.mark.parametrize(
    "mask, clean, ids",
    [
        (_three_objects((1, 17, 5)), [1, 2, 3], [1, 5, 17]),  # gapped, non-monotonic
        (_M3D, [1, 2], [3, 9]),  # 3D
        (np.zeros((8, 8), np.int32), [], []),  # all background
        (np.array([[False, True], [True, False]]), [1], [1]),  # bool -> one object
    ],
)
def test_sanitize_relabels(mask, clean, ids):
    out, out_ids = sanitize_masks(mask)
    assert sorted(set(np.unique(out)) - {0}) == clean
    assert out_ids.tolist() == ids


def test_no_mutation_and_copy_policy():
    contig, gapped = _three_objects((1, 2, 3)), _three_objects((1, 17, 5))
    assert sanitize_masks(contig)[0] is contig  # already 1..N: no copy
    before = gapped.copy()
    assert sanitize_masks(gapped)[0] is not gapped  # relabelled: fresh copy
    assert np.array_equal(gapped, before)  # input untouched


@pytest.mark.parametrize("bad", [np.array([[0, -1, 2]]), np.array([[1.0, 2.0]])])
def test_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        sanitize_masks(bad)


def _is_sanitized(fn):
    return bool(
        getattr(fn, "_sanitized", False)
        or getattr(getattr(fn, "func", None), "_sanitized", False)
    )


def test_public_funcs_sanitized_multimask_excluded():
    public = (
        *get_core_measurements().values(),
        *get_core_measurements(legacy=True).values(),  # partial-wrapped intensity
        *get_correlation_measurements().values(),
        get_correlation_overlap,  # public, not in the registry
    )
    assert all(_is_sanitized(fn) for fn in public)
    # two-mask functions are out of scope and must stay unwrapped
    assert not any(_is_sanitized(fn) for fn in get_multimask_measurements().values())


def test_decorated_func_handles_gapped_labels():
    # get_zernike raises on gapped labels without sanitation; the decorator fixes it.
    px = np.random.default_rng(0).random((64, 64))
    zernike = get_core_measurements()["zernike"]
    gapped = zernike(_three_objects((1, 17, 5)), px)
    contig = zernike(_three_objects((1, 3, 2)), px)
    for key in gapped:
        np.testing.assert_allclose(gapped[key], contig[key], equal_nan=True)


def test_featurizer_uses_original_ids_and_sanitizes_once(monkeypatch):
    import cp_measure.featurizer as fz

    calls = []
    real = fz.sanitize_masks
    monkeypatch.setattr(fz, "sanitize_masks", lambda m: (calls.append(1), real(m))[1])
    img = np.random.default_rng(0).random((2, 64, 64))
    config = make_featurizer_config(["DNA", "ER"])

    data_g, _, rows_g = featurize(img, _three_objects((1, 17, 5))[np.newaxis], config)
    assert len(calls) == 1  # cost paid once per object-type mask
    data_c, _, _ = featurize(img, _three_objects((1, 3, 2))[np.newaxis], config)
    # original IDs in the rows, and identical geometry -> identical values
    assert rows_g == [(None, "object", 1), (None, "object", 5), (None, "object", 17)]
    np.testing.assert_allclose(data_g, data_c, equal_nan=True)
