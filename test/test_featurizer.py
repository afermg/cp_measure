"""Tests for the featurizer wrapper.

All inputs use the canonical batch contract ``(B, C, *spatial)`` / ``(B, M, *spatial)``.
These single-image cases use ``B == 1``; batch-specific behaviour (stacking, empty-image
skipping, ``is_3d`` rank validation) lives in ``test_featurizer_batch.py``.
"""

import numpy as np
import pytest

from cp_measure.featurizer import featurize, make_featurizer_config

from conftest import ALL_OFF, CELL_PAINTING_CHANNELS, SIZE_2D, get_rng


def _b(*arrays):
    """Add a unit batch axis to each array (single-image -> B=1 batch)."""
    return tuple(a[None] for a in arrays)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestSmoke:
    """High-level test to capture general misbehaviour."""

    def test_default_config(self, image_2d_2ch, mask_2d):
        with pytest.warns(UserWarning, match="No channel names"):
            data, columns, rows = featurize(*_b(image_2d_2ch, mask_2d))

        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert data.shape[0] == 2
        assert data.shape[1] == len(columns)
        assert len(rows) == 2
        assert rows[0] == (0, "object", 1)
        assert rows[1] == (0, "object", 2)

    def test_custom_config(self, image_2d_2ch, mask_2d):
        config = make_featurizer_config(
            ["DNA", "ER"], **{**ALL_OFF, "intensity": True, "sizeshape": True}
        )
        data, columns, rows = featurize(*_b(image_2d_2ch, mask_2d), config)

        assert data.shape[0] == 2
        assert data.shape[1] == len(columns)
        assert any("__DNA" in c for c in columns)
        assert any("__ER" in c for c in columns)

    def test_image_id_propagated(self, image_2d_1ch, mask_2d):
        config = make_featurizer_config(["DNA"], **{**ALL_OFF, "intensity": True})
        _, _, rows = featurize(
            *_b(image_2d_1ch, mask_2d), config, image_ids=["plate1_A01"]
        )
        assert all(r[0] == "plate1_A01" for r in rows)

    def test_values_finite_and_nontrivial(self, image_2d_2ch, mask_2d):
        with pytest.warns(UserWarning, match="No channel names"):
            data, columns, _ = featurize(*_b(image_2d_2ch, mask_2d))

        nonzero_frac = np.mean(np.any(data != 0, axis=0))
        assert nonzero_frac > 0.5
        assert not np.any(np.isinf(data))

    def test_correlation_features(self, image_2d_2ch, mask_2d):
        config = make_featurizer_config(
            ["DNA", "ER"],
            **{**ALL_OFF, "intensity": True, "correlation_pearson": True},
        )
        data, columns, rows = featurize(*_b(image_2d_2ch, mask_2d), config)
        corr_cols = [c for c in columns if "Correlation" in c or "Slope" in c]
        assert len(corr_cols) > 0, "Expected correlation columns"
        assert data.shape[0] == 2

    def test_3d_single_channel(self, image_3d_1ch, mask_3d):
        config = make_featurizer_config(
            ["DNA"], **{**ALL_OFF, "intensity": True, "sizeshape": True}
        )
        data, columns, rows = featurize(*_b(image_3d_1ch, mask_3d), config, is_3d=True)
        assert data.shape[0] == 2
        assert any("Intensity" in c for c in columns)
        assert any("Area" in c for c in columns)

    def test_3d_two_channels(self, image_3d_2ch, mask_3d):
        config = make_featurizer_config(
            CELL_PAINTING_CHANNELS[:2],
            **{
                **ALL_OFF,
                "intensity": True,
                "sizeshape": True,
                "correlation_pearson": True,
            },
        )
        data, columns, rows = featurize(*_b(image_3d_2ch, mask_3d), config, is_3d=True)
        assert data.shape[0] == 2
        assert data.shape[1] == len(columns)
        assert any("__DNA" in c for c in columns)
        assert any("__ER" in c for c in columns)
        assert any("Correlation" in c or "Slope" in c for c in columns)

    def test_3d_skips_2d_only_features(self, image_3d_2ch, mask_3d):
        config = make_featurizer_config(
            ["DNA", "ER"],
            **{**ALL_OFF, "intensity": True, "sizeshape": True, "zernike": True},
        )
        data, columns, rows = featurize(*_b(image_3d_2ch, mask_3d), config, is_3d=True)

        assert data.shape[0] == 2
        assert not any("Zernike" in c for c in columns)
        assert any("Intensity" in c for c in columns)
        assert any("Area" in c for c in columns)


# ---------------------------------------------------------------------------
# Channel auto-naming
# ---------------------------------------------------------------------------


class TestChannelAutoNaming:
    def test_warns_when_no_channels(self, image_2d_2ch, mask_2d):
        config = make_featurizer_config(**{**ALL_OFF, "intensity": True})
        with pytest.warns(UserWarning, match="No channel names"):
            data, columns, rows = featurize(*_b(image_2d_2ch, mask_2d), config)
        assert any("__ch0" in c for c in columns)
        assert any("__ch1" in c for c in columns)

    def test_zero_padded_when_many_channels(self, mask_2d):
        image = get_rng().random((12, SIZE_2D, SIZE_2D))
        config = make_featurizer_config(**{**ALL_OFF, "intensity": True})
        with pytest.warns(UserWarning, match="No channel names"):
            _, columns, _ = featurize(*_b(image, mask_2d), config)
        assert any("__ch00" in c for c in columns)
        assert any("__ch11" in c for c in columns)


# ---------------------------------------------------------------------------
# Parameter forwarding
# ---------------------------------------------------------------------------


class TestParamsForwarding:
    def test_granularity_spectrum_length(self, image_2d_1ch, mask_2d):
        c4 = make_featurizer_config(
            ["DNA"],
            **{**ALL_OFF, "granularity": True},
            granularity_params={"granular_spectrum_length": 4},
        )
        c8 = make_featurizer_config(
            ["DNA"],
            **{**ALL_OFF, "granularity": True},
            granularity_params={"granular_spectrum_length": 8},
        )
        _, cols4, _ = featurize(*_b(image_2d_1ch, mask_2d), c4)
        _, cols8, _ = featurize(*_b(image_2d_1ch, mask_2d), c8)

        assert len(cols8) > len(cols4)


# ---------------------------------------------------------------------------
# Per-image validation (routed through the per-item worker)
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        "config, image, masks, match",
        [
            (
                make_featurizer_config(["DNA", "ER"], **{**ALL_OFF, "intensity": True}),
                np.ones((1, 3, 10, 10)),
                np.ones((1, 1, 10, 10), dtype=np.int32),
                "channels",
            ),
            (
                make_featurizer_config(
                    ["DNA"], objects=["nuclei"], **{**ALL_OFF, "intensity": True}
                ),
                np.ones((1, 1, 10, 10)),
                np.ones((1, 2, 10, 10), dtype=np.int32),
                "object names",
            ),
        ],
    )
    def test_name_count_mismatch(self, config, image, masks, match):
        with pytest.raises(ValueError, match=match):
            featurize(image, masks, config)


# ---------------------------------------------------------------------------
# Factory validation
# ---------------------------------------------------------------------------


class TestMakeFeaturizer:
    def test_no_features_raises(self):
        with pytest.raises(ValueError, match="at least one feature"):
            make_featurizer_config(["DNA"], **ALL_OFF)

    def test_empty_channels_with_shape_only(self):
        config = make_featurizer_config(
            [], **{**ALL_OFF, "sizeshape": True, "zernike": True, "feret": True}
        )
        assert config["channels"] == []
        assert config["sizeshape"] is True

    def test_empty_objects_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_featurizer_config(
                ["DNA"], objects=[], **{**ALL_OFF, "intensity": True}
            )

    def test_duplicate_channels_raises(self):
        with pytest.raises(ValueError, match="unique"):
            make_featurizer_config(["DNA", "DNA"], **{**ALL_OFF, "intensity": True})

    def test_duplicate_objects_raises(self):
        with pytest.raises(ValueError, match="unique"):
            make_featurizer_config(
                ["DNA"],
                objects=["cells", "cells"],
                **{**ALL_OFF, "intensity": True},
            )

    def test_single_channel_correlation_warns(self, image_2d_1ch, mask_2d):
        config = make_featurizer_config(
            ["DNA"], **{**ALL_OFF, "intensity": True, "correlation_pearson": True}
        )
        with pytest.warns(UserWarning, match="at least 2 channels"):
            data, columns, _ = featurize(*_b(image_2d_1ch, mask_2d), config)
        assert not any("Correlation" in c for c in columns)


# ---------------------------------------------------------------------------
# Multi-mask
# ---------------------------------------------------------------------------


class TestMultiMask:
    def test_multi_mask_stacks_rows(self, image_2d_2ch, masks_2d_multi):
        config = make_featurizer_config(
            CELL_PAINTING_CHANNELS[:2],
            objects=["nuclei", "cells"],
            **{**ALL_OFF, "intensity": True, "sizeshape": True},
        )
        data, columns, rows = featurize(*_b(image_2d_2ch, masks_2d_multi), config)

        assert data.shape[0] == 5
        assert len(rows) == 5
        assert rows[0] == (0, "nuclei", 1)
        assert rows[1] == (0, "nuclei", 2)
        assert rows[2] == (0, "cells", 1)
        assert rows[3] == (0, "cells", 2)
        assert rows[4] == (0, "cells", 3)
        assert data.shape[1] == len(columns)

    @pytest.mark.parametrize(
        "image_fixture,channels",
        [
            ("image_3d_2ch", CELL_PAINTING_CHANNELS[:2]),
            ("image_3d_1ch", ["DNA"]),
        ],
    )
    def test_3d_multi_mask(self, request, image_fixture, channels, masks_3d_multi):
        image = request.getfixturevalue(image_fixture)
        config = make_featurizer_config(
            channels,
            objects=["nuclei", "cells"],
            **{**ALL_OFF, "intensity": True, "sizeshape": True},
        )
        data, columns, rows = featurize(*_b(image, masks_3d_multi), config, is_3d=True)

        assert data.shape[0] == 3
        assert len(rows) == 3
        assert rows[0] == (0, "nuclei", 1)
        assert rows[1] == (0, "cells", 1)
        assert rows[2] == (0, "cells", 2)
        assert data.shape[1] == len(columns)

    def test_empty_object_mask_skipped(self, image_2d_1ch):
        """One empty object-mask among several is skipped (within a non-empty image)."""
        mask1 = np.zeros((SIZE_2D, SIZE_2D), dtype=np.int32)
        mask1[5:15, 5:15] = 1
        mask2 = np.zeros((SIZE_2D, SIZE_2D), dtype=np.int32)

        masks = np.stack([mask1, mask2], axis=0)
        config = make_featurizer_config(
            ["DNA"],
            objects=["nuclei", "cells"],
            **{**ALL_OFF, "intensity": True},
        )
        data, columns, rows = featurize(*_b(image_2d_1ch, masks), config)

        assert data.shape[0] == 1
        assert rows == [(0, "nuclei", 1)]
