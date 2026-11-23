"""Costes fits each pair once by default without changing other correlations."""

import itertools

import numpy as np
import pytest

import cp_measure
from cp_measure import bulk
from cp_measure._detect import HAS_NUMBA
from cp_measure._sanitize import sanitize_masks
from cp_measure.featurizer import featurize, make_featurizer_config

from conftest import ALL_OFF

BACKENDS = [
    pytest.param(None, id="numpy"),
    pytest.param(
        "numba", marks=pytest.mark.skipif(not HAS_NUMBA, reason="needs numba")
    ),
]
CHANNELS = ["Zeta", "Alpha", "Mu"]  # Orientation follows axis order, not sorting.
OBJECTS = ["cells", "nuclei"]
KEYS = {
    "pearson": ["Correlation_Pearson", "Correlation_Slope"],
    "costes": ["Correlation_Costes_1", "Correlation_Costes_2"],
    "manders_fold": ["Correlation_Manders_1", "Correlation_Manders_2"],
    "rwc": ["Correlation_RWC_1", "Correlation_RWC_2"],
}


def config(**kwargs):
    options = {
        **ALL_OFF,
        **{f"correlation_{name}": True for name in KEYS},
        "objects": OBJECTS,
    }
    options.update(kwargs)
    return make_featurizer_config(CHANNELS, **options)


def inputs(is_3d=False):
    rng = np.random.default_rng(917)
    base = rng.integers(5, 160, size=(2, 1, 12, 13))
    image = np.concatenate(
        [base + rng.integers(0, 55, size=base.shape) for _ in CHANNELS], axis=1
    ).astype(np.uint8)
    masks = np.zeros((2, 2, 12, 13), dtype=np.int32)
    masks[:, :, 1:6, 1:6] = 7
    masks[:, :, 6:11, 7:12] = 42
    masks[1, 1] = 0
    if is_3d:
        image = np.repeat(image[:, :, None], 2, axis=2)
        masks = np.repeat(masks[:, :, None], 2, axis=2)
    return image, masks


def pairs_for(name, mode):
    ordered = name == "pearson" or (name == "costes" and mode == "ordered")
    iterator = itertools.permutations if ordered else itertools.combinations
    return list(iterator(range(len(CHANNELS)), 2))


@pytest.mark.parametrize("accelerator", BACKENDS)
@pytest.mark.parametrize("mode", ["unordered", "ordered"])
@pytest.mark.parametrize("scope", ["object", "image"])
@pytest.mark.parametrize("precision", ["input", "float64"])
@pytest.mark.parametrize("is_3d", [False, True])
def test_execution_and_direct_values(
    monkeypatch, accelerator, mode, scope, precision, is_3d
):
    monkeypatch.setattr(cp_measure, "_ACCELERATOR", accelerator)
    funcs = bulk.get_correlation_measurements(sanitize=False)
    calls = {name: [] for name in KEYS}

    def spy(name):
        def measure(*, masks, pixels_1, pixels_2, **kwargs):
            calls[name].append((masks, pixels_1, pixels_2, kwargs))
            return funcs[name](
                masks=masks, pixels_1=pixels_1, pixels_2=pixels_2, **kwargs
            )

        return measure

    monkeypatch.setattr(
        bulk, "get_correlation_measurements", lambda **_: {n: spy(n) for n in KEYS}
    )
    image, masks = inputs(is_3d)
    before_image, before_masks = image.copy(), masks.copy()
    cfg = config(
        costes_pair_mode=mode,
        correlation_costes_params={
            "scope": scope,
            "precision": precision,
            "scale_max": 255,
        },
        correlation_manders_fold_params={"thr": 17},
        correlation_rwc_params={"thr": 13, "rank_scope": "object_set"},
    )
    actual, columns, rows = featurize(
        image, masks, cfg, image_ids=["fov-z", "fov-a"], is_3d=is_3d
    )
    assert columns == [
        f"{key}__{CHANNELS[a]}__{CHANNELS[b]}"
        for name in KEYS
        for a, b in pairs_for(name, mode)
        for key in KEYS[name]
    ]
    assert len(columns) == len(set(columns))
    assert rows == [
        (fov, obj, label)
        for fov, obj in [("fov-z", "cells"), ("fov-z", "nuclei"), ("fov-a", "cells")]
        for label in (7, 42)
    ]
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(masks, before_masks)
    for name in KEYS:
        pairs = pairs_for(name, mode)
        params = cfg.get(f"correlation_{name}_params", {})
        assert len(calls[name]) == 2  # One batched invocation per mask type.
        assert sum(len(call[0]) for call in calls[name]) == 3 * len(pairs)
        for mask_index, (clean_masks, first, second, kwargs) in enumerate(calls[name]):
            assert kwargs == params
            fovs = [0, 1] if mask_index == 0 else [0]
            work = [(fov, a, b) for fov in fovs for a, b in pairs]
            assert len(clean_masks) == len(work)
            for k, (fov, a, b) in enumerate(work):
                clean, ids = sanitize_masks(masks[fov, mask_index])
                np.testing.assert_array_equal(clean_masks[k], clean)
                assert np.shares_memory(first[k], image[fov, a])
                assert np.shares_memory(second[k], image[fov, b])
                direct = funcs[name](
                    masks=clean,
                    pixels_1=image[fov, a],
                    pixels_2=image[fov, b],
                    **params,
                )
                row_indices = [
                    rows.index((["fov-z", "fov-a"][fov], OBJECTS[mask_index], label))
                    for label in ids
                ]
                for key in KEYS[name]:
                    column = columns.index(f"{key}__{CHANNELS[a]}__{CHANNELS[b]}")
                    np.testing.assert_allclose(
                        actual[row_indices, column],
                        direct[key],
                        rtol=1e-10,
                        atol=1e-12,
                        equal_nan=True,
                    )


def test_default_keeps_both_forward_coefficients_without_reverse_calls(monkeypatch):
    calls = []

    def costes(*, masks, pixels_1, pixels_2, **kwargs):
        calls.extend(
            (int(a.flat[0]), int(b.flat[0])) for a, b in zip(pixels_1, pixels_2)
        )
        return [
            {
                "Correlation_Costes_1": np.array([a.flat[0] * 10 + b.flat[0]]),
                "Correlation_Costes_2": np.array([a.flat[0] * 100 + b.flat[0]]),
            }
            for a, b in zip(pixels_1, pixels_2)
        ]

    monkeypatch.setattr(
        bulk, "get_correlation_measurements", lambda **_: {"costes": costes}
    )
    image = np.broadcast_to(np.arange(1, 4)[None, :, None, None], (1, 3, 3, 3))
    masks = np.ones((1, 1, 3, 3), np.int32)
    cfg = config(
        objects=["cells"],
        correlation_pearson=False,
        correlation_manders_fold=False,
        correlation_rwc=False,
    )
    assert cfg["costes_pair_mode"] == "unordered"
    values, columns, _ = featurize(image, masks, cfg)
    assert calls == [(1, 2), (1, 3), (2, 3)]
    assert columns == [
        f"Correlation_Costes_{coefficient}__{CHANNELS[a]}__{CHANNELS[b]}"
        for a, b in [(0, 1), (0, 2), (1, 2)]
        for coefficient in (1, 2)
    ]
    np.testing.assert_array_equal(values, [[12, 102, 13, 103, 23, 203]])


@pytest.mark.parametrize("accelerator", BACKENDS)
def test_new_default_is_exact_subset_of_ordered_results(monkeypatch, accelerator):
    monkeypatch.setattr(cp_measure, "_ACCELERATOR", accelerator)
    image, masks = inputs()
    cfg = config()
    values, columns, rows = featurize(image, masks, cfg)
    old, old_columns, old_rows = featurize(
        image, masks, {**cfg, "costes_pair_mode": "ordered"}
    )
    assert rows == old_rows
    np.testing.assert_array_equal(
        values, old[:, [old_columns.index(c) for c in columns]]
    )
    assert "Correlation_Slope__Mu__Zeta" in columns
    assert "Correlation_Pearson__Mu__Zeta" in columns
    assert "Correlation_Costes_2__Mu__Zeta" not in columns
    assert "Correlation_Costes_2__Mu__Zeta" in old_columns
    assert len(columns) == 30 and len(old_columns) == 36
    del cfg["costes_pair_mode"]
    missing, missing_columns, missing_rows = featurize(image, masks, cfg)
    assert columns == missing_columns and rows == missing_rows
    np.testing.assert_array_equal(values, missing)


@pytest.mark.parametrize("mode", [None, "", "cellprofiler", "typo", 1, True, [], {}])
def test_invalid_mode_builder(mode):
    with pytest.raises(ValueError, match="costes_pair_mode"):
        config(costes_pair_mode=mode)


@pytest.mark.parametrize("nchannels", [1, 3])
@pytest.mark.parametrize("mode", [None, "typo", []])
def test_invalid_plain_config_even_with_costes_disabled(mode, nchannels):
    image, masks = inputs()
    cfg = config(**{f"correlation_{name}": False for name in KEYS}, intensity=True)
    cfg["costes_pair_mode"] = mode
    cfg["channels"] = CHANNELS[:nchannels]
    with pytest.raises(ValueError, match="costes_pair_mode"):
        featurize(image[:, :nchannels], masks, cfg)


@pytest.mark.parametrize("accelerator", BACKENDS)
def test_empty_fov_and_auto_names(monkeypatch, accelerator):
    monkeypatch.setattr(cp_measure, "_ACCELERATOR", accelerator)
    image, masks = inputs()
    masks[0] = 0
    cfg = config()
    cfg["channels"] = None
    with pytest.warns(UserWarning) as caught:
        values, columns, rows = featurize(image, masks, cfg)
    assert any("auto-assigning" in str(w.message) for w in caught)
    assert any("batch indices [0]" in str(w.message) for w in caught)
    assert rows == [(1, "cells", 7), (1, "cells", 42)]
    assert values.shape == (2, 30)
    assert "Correlation_Slope__ch2__ch0" in columns
    assert "Correlation_Costes_2__ch2__ch0" not in columns
