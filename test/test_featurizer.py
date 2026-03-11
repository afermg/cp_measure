"""Tests for the featurizer wrapper."""

import numpy as np
import pytest

from cp_measure.featurizer import featurize, make_featurizer

CELL_PAINTING_CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]

# All feature flags set to False — tests override only what they need.
_ALL_OFF = dict(
    intensity=False,
    texture=False,
    granularity=False,
    radial_distribution=False,
    radial_zernikes=False,
    sizeshape=False,
    zernike=False,
    ferret=False,
    correlation_pearson=False,
    correlation_costes=False,
    correlation_manders_fold=False,
    correlation_rwc=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_and_mask(n_channels=2, size=64, n_objects=2, seed=42, dtype=np.float64):
    """Create a small random image ``(C, H, W)`` and mask ``(1, H, W)``."""
    rng = np.random.default_rng(seed)
    image = rng.random((n_channels, size, size)).astype(dtype)
    mask = np.zeros((1, size, size), dtype=np.int32)
    step = size // (n_objects + 1)
    obj_size = max(step // 2, 8)
    for i in range(n_objects):
        r = step * (i + 1) - obj_size // 2
        c = step * (i + 1) - obj_size // 2
        mask[0, r : r + obj_size, c : c + obj_size] = i + 1
    return image, mask


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestSmoke:
    """High-level test to capture general misbehaviour."""
    def test_intensity_and_sizeshape(self):
        channels = CELL_PAINTING_CHANNELS[:2]
        image, mask = _make_image_and_mask(n_channels=2)
        config = make_featurizer(
            channels, **{**_ALL_OFF, "intensity": True, "sizeshape": True}
        )
        data, columns, rows = featurize(image, mask, config)

        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert data.shape[0] == 2  # 2 objects
        assert data.shape[1] == len(columns)
        assert len(rows) == 2
        # Row tuples have shape (image_id, object, label)
        assert rows[0] == (None, "object", 1)
        assert rows[1] == (None, "object", 2)
        # Channel names appear in column names
        assert any("__DNA" in c for c in columns)
        assert any("__ER" in c for c in columns)

    def test_single_feature(self):
        image, mask = _make_image_and_mask(n_channels=1)
        config = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        data, columns, rows = featurize(image, mask, config)
        assert data.shape[0] == 2
        assert len(columns) > 0

    def test_image_id_propagated(self):
        image, mask = _make_image_and_mask(n_channels=1)
        config = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        _, _, rows = featurize(image, mask, config, image_id="plate1_A01")
        assert all(r[0] == "plate1_A01" for r in rows)

    def test_values_finite_and_nontrivial(self):
        channels = CELL_PAINTING_CHANNELS[:2]
        image, mask = _make_image_and_mask(n_channels=2)
        config = make_featurizer(
            channels, **{**_ALL_OFF, "intensity": True, "sizeshape": True}
        )
        data, columns, _ = featurize(image, mask, config)

        # No column should be entirely zero
        nonzero_frac = np.mean(np.any(data != 0, axis=0))
        assert nonzero_frac > 0.5

        # No infs
        assert not np.any(np.isinf(data))

    def test_correlation_features(self):
        image, mask = _make_image_and_mask(n_channels=2)
        config = make_featurizer(
            ["DNA", "ER"],
            **{**_ALL_OFF, "intensity": True, "correlation_pearson": True},
        )
        data, columns, rows = featurize(image, mask, config)
        corr_cols = [c for c in columns if "Correlation" in c or "Slope" in c]
        assert len(corr_cols) > 0, "Expected correlation columns"
        assert data.shape[0] == 2


# ---------------------------------------------------------------------------
# Channel auto-naming
# ---------------------------------------------------------------------------


class TestChannelAutoNaming:
    def test_warns_when_no_channels(self):
        image, mask = _make_image_and_mask(n_channels=2)
        config = make_featurizer(**{**_ALL_OFF, "intensity": True})
        with pytest.warns(UserWarning, match="No channel names"):
            data, columns, rows = featurize(image, mask, config)
        assert any("__ch0" in c for c in columns)
        assert any("__ch1" in c for c in columns)

    def test_zero_padded_when_many_channels(self):
        image, mask = _make_image_and_mask(n_channels=12, size=64)
        config = make_featurizer(**{**_ALL_OFF, "intensity": True})
        with pytest.warns(UserWarning, match="No channel names"):
            _, columns, _ = featurize(image, mask, config)
        assert any("__ch00" in c for c in columns)
        assert any("__ch11" in c for c in columns)


# ---------------------------------------------------------------------------
# Parameter forwarding
# ---------------------------------------------------------------------------


class TestParamsForwarding:
    def test_granularity_spectrum_length(self):
        image, mask = _make_image_and_mask(n_channels=1)

        c4 = make_featurizer(
            ["DNA"],
            **{**_ALL_OFF, "granularity": True},
            granularity_params={"granular_spectrum_length": 4},
        )
        c8 = make_featurizer(
            ["DNA"],
            **{**_ALL_OFF, "granularity": True},
            granularity_params={"granular_spectrum_length": 8},
        )
        _, cols4, _ = featurize(image, mask, c4)
        _, cols8, _ = featurize(image, mask, c8)

        assert len(cols8) > len(cols4)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_image_2d_raises(self):
        config = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="3D.*4D"):
            featurize(np.ones((10, 10)), np.ones((1, 10, 10), dtype=np.int32), config)

    def test_ndim_mismatch(self):
        config = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="same number of dimensions"):
            featurize(
                np.ones((1, 10, 10)), np.ones((1, 2, 10, 10), dtype=np.int32), config
            )

    def test_channel_count_mismatch(self):
        config = make_featurizer(["DNA", "ER"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="channels"):
            featurize(
                np.ones((3, 10, 10)), np.ones((1, 10, 10), dtype=np.int32), config
            )

    def test_mask_count_mismatch(self):
        config = make_featurizer(
            ["DNA"], objects=["nuclei"], **{**_ALL_OFF, "intensity": True}
        )
        with pytest.raises(ValueError, match="object names"):
            featurize(
                np.ones((1, 10, 10)), np.ones((2, 10, 10), dtype=np.int32), config
            )

    def test_spatial_dims_mismatch(self):
        config = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="spatial dims"):
            featurize(np.ones((1, 10, 10)), np.ones((1, 8, 8), dtype=np.int32), config)

    def test_mask_not_integer(self):
        config = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(TypeError, match="integer dtype"):
            featurize(
                np.ones((1, 10, 10)), np.ones((1, 10, 10), dtype=np.float64), config
            )

    def test_all_masks_empty(self):
        config = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="no labels"):
            featurize(
                np.ones((1, 10, 10)), np.zeros((1, 10, 10), dtype=np.int32), config
            )


# ---------------------------------------------------------------------------
# Factory validation
# ---------------------------------------------------------------------------


class TestMakeFeaturizer:
    def test_no_features_raises(self):
        with pytest.raises(ValueError, match="at least one feature"):
            make_featurizer(["DNA"], **_ALL_OFF)

    def test_empty_channels_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_featurizer([], **{**_ALL_OFF, "intensity": True})

    def test_empty_objects_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_featurizer(["DNA"], objects=[], **{**_ALL_OFF, "intensity": True})

    def test_duplicate_channels_raises(self):
        with pytest.raises(ValueError, match="unique"):
            make_featurizer(["DNA", "DNA"], **{**_ALL_OFF, "intensity": True})

    def test_duplicate_objects_raises(self):
        with pytest.raises(ValueError, match="unique"):
            make_featurizer(
                ["DNA"],
                objects=["cells", "cells"],
                **{**_ALL_OFF, "intensity": True},
            )

    def test_single_channel_correlation_warns(self):
        config = make_featurizer(
            ["DNA"], **{**_ALL_OFF, "intensity": True, "correlation_pearson": True}
        )
        image, mask = _make_image_and_mask(n_channels=1)
        with pytest.warns(UserWarning, match="at least 2 channels"):
            data, columns, _ = featurize(image, mask, config)
        assert not any("Correlation" in c for c in columns)


# ---------------------------------------------------------------------------
# Multi-mask
# ---------------------------------------------------------------------------


class TestMultiMask:
    def test_multi_mask_stacks_rows(self):
        rng = np.random.default_rng(42)
        size = 64
        image = rng.random((2, size, size))

        nuclei = np.zeros((size, size), dtype=np.int32)
        nuclei[5:15, 5:15] = 1
        nuclei[30:40, 30:40] = 2

        cells = np.zeros((size, size), dtype=np.int32)
        cells[3:18, 3:18] = 1
        cells[28:45, 28:45] = 2
        cells[50:60, 50:60] = 3

        masks = np.stack([nuclei, cells], axis=0)
        config = make_featurizer(
            CELL_PAINTING_CHANNELS[:2],
            objects=["nuclei", "cells"],
            **{**_ALL_OFF, "intensity": True, "sizeshape": True},
        )
        data, columns, rows = featurize(image, masks, config)

        # 2 nuclei + 3 cells = 5 rows
        assert data.shape[0] == 5
        assert len(rows) == 5
        assert rows[0] == (None, "nuclei", 1)
        assert rows[1] == (None, "nuclei", 2)
        assert rows[2] == (None, "cells", 1)
        assert rows[3] == (None, "cells", 2)
        assert rows[4] == (None, "cells", 3)
        # All rows share the same columns
        assert data.shape[1] == len(columns)

    def test_empty_mask_skipped(self):
        rng = np.random.default_rng(42)
        size = 64
        image = rng.random((1, size, size))

        mask1 = np.zeros((size, size), dtype=np.int32)
        mask1[5:15, 5:15] = 1
        mask2 = np.zeros((size, size), dtype=np.int32)

        masks = np.stack([mask1, mask2], axis=0)
        config = make_featurizer(
            ["DNA"],
            objects=["nuclei", "cells"],
            **{**_ALL_OFF, "intensity": True},
        )
        data, columns, rows = featurize(image, masks, config)

        assert data.shape[0] == 1
        assert rows == [(None, "nuclei", 1)]
