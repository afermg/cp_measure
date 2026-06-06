"""The fused numba colocalization producer (`get_correlation_all`).

`get_correlation_all(features=None)` returns every coloc feature from ONE flatten + ONE
`coloc_per_object` pass; with a `features` subset it returns exactly those groups, gating RWC's
rank sort and the Costes kernel to what was requested. The five single-feature functions are thin
gated wrappers over it. The numba correlation registry keeps per-group keys (so featurize's
per-group selection works) and `featurize` runs under the numba accelerator.
"""

import numpy as np
import pytest
from conftest import (
    DEPTH_3D,
    SIZE_2D,
    SIZE_3D,
    _stamp_objects_2d,
    _stamp_objects_3d,
    get_rng,
)

import cp_measure.core.measurecolocalization as ref
from cp_measure._detect import HAS_NUMBA

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")

# group -> the wrapper function name and the feature keys it produces
GROUP_KEYS = {
    "pearson": (ref.F_CORRELATION_FORMAT, ref.F_SLOPE_FORMAT),
    "manders_fold": (f"{ref.F_MANDERS_FORMAT}_1", f"{ref.F_MANDERS_FORMAT}_2"),
    "overlap": (ref.F_OVERLAP_FORMAT, f"{ref.F_K_FORMAT}_1", f"{ref.F_K_FORMAT}_2"),
    "rwc": (f"{ref.F_RWC_FORMAT}_1", f"{ref.F_RWC_FORMAT}_2"),
    "costes": (f"{ref.F_COSTES_FORMAT}_1", f"{ref.F_COSTES_FORMAT}_2"),
}


def _nb():
    import cp_measure.core.numba.measurecolocalization as nb

    return nb


def _data_2d():
    masks = np.zeros((SIZE_2D, SIZE_2D), np.int32)
    _stamp_objects_2d(masks, n_objects=3)
    rng = get_rng()
    return masks, rng.random(masks.shape), rng.random(masks.shape)


def _data_3d():
    masks = np.zeros((DEPTH_3D, SIZE_3D, SIZE_3D), np.int32)
    _stamp_objects_3d(masks, n_objects=2)
    rng = get_rng()
    return masks, rng.random(masks.shape), rng.random(masks.shape)


def _union_separate(p1, p2, masks):
    nb = _nb()
    out = {}
    for name in GROUP_KEYS:
        out.update(getattr(nb, f"get_correlation_{name}")(p1, p2, masks))
    return out


@requires_numba
@pytest.mark.parametrize("data", [_data_2d, _data_3d])
def test_fused_all_is_bit_identical_to_separate(data):
    masks, p1, p2 = data()
    fused = _nb().get_correlation_all(p1, p2, masks)  # features=None -> all
    sep = _union_separate(p1, p2, masks)
    assert set(fused) == set(sep), set(fused).symmetric_difference(sep)
    for key in sep:
        np.testing.assert_array_equal(fused[key], sep[key], err_msg=f"feature {key!r}")


@requires_numba
@pytest.mark.parametrize("group", list(GROUP_KEYS))
def test_subset_returns_only_requested_and_matches_wrapper(group):
    masks, p1, p2 = _data_2d()
    sub = _nb().get_correlation_all(p1, p2, masks, features=[group])
    assert set(sub) == set(GROUP_KEYS[group])
    # the single-feature wrapper is exactly this subset
    wrapper = getattr(_nb(), f"get_correlation_{group}")(p1, p2, masks)
    assert set(wrapper) == set(sub)
    for key in sub:
        np.testing.assert_array_equal(wrapper[key], sub[key], err_msg=key)


@requires_numba
def test_multi_subset_returns_exactly_those_groups():
    masks, p1, p2 = _data_2d()
    out = _nb().get_correlation_all(p1, p2, masks, features=["pearson", "rwc"])
    expected = set(GROUP_KEYS["pearson"]) | set(GROUP_KEYS["rwc"])
    assert set(out) == expected
    full = _nb().get_correlation_all(p1, p2, masks)
    for key in expected:  # values identical to the full run
        np.testing.assert_array_equal(out[key], full[key], err_msg=key)


@requires_numba
def test_unknown_feature_raises():
    masks, p1, p2 = _data_2d()
    with pytest.raises(ValueError, match="unknown correlation feature"):
        _nb().get_correlation_all(p1, p2, masks, features=["bogus"])


@requires_numba
def test_empty_mask_returns_empty_arrays():
    masks = np.zeros((SIZE_2D, SIZE_2D), np.int32)
    rng = get_rng()
    out = _nb().get_correlation_all(
        rng.random(masks.shape), rng.random(masks.shape), masks
    )
    for key, val in out.items():
        assert val.shape == (0,), key


@requires_numba
def test_batch_list_matches_per_image():
    imgs = [_data_2d(), _data_3d()]
    masks = [m for m, _, _ in imgs]
    p1 = [a for _, a, _ in imgs]
    p2 = [b for _, _, b in imgs]
    got = _nb().get_correlation_all(p1, p2, masks)
    assert isinstance(got, list) and len(got) == 2
    for (m, a, b), per_image in zip(imgs, got):
        sep = _union_separate(a, b, m)
        assert set(per_image) == set(sep)
        for key in sep:
            np.testing.assert_array_equal(per_image[key], sep[key], err_msg=key)


@requires_numba
def test_registry_keeps_per_group_keys_and_featurize_runs():
    import cp_measure
    from cp_measure.bulk import get_correlation_measurements
    from cp_measure.featurizer import featurize

    cp_measure.set_accelerator("numba")
    try:
        reg = get_correlation_measurements()
        assert {"pearson", "manders_fold", "rwc", "costes", "overlap"} <= set(reg)
        # featurize must not KeyError under numba and must emit correlation columns
        rng = get_rng()
        image = rng.random((2, SIZE_2D, SIZE_2D))
        masks = np.zeros((1, SIZE_2D, SIZE_2D), np.int32)
        _stamp_objects_2d(masks[0], n_objects=3)
        _, columns, _ = featurize(image, masks)
        assert any("Correlation" in c for c in columns)
    finally:
        cp_measure.set_accelerator(None)
