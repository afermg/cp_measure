"""Tests for the high-level Featurizer orchestrator."""

import itertools

import numpy as np
import pandas as pd
import pytest

from cp_measure.featurizer import make_featurizer

# Standard Cell Painting assay channels (stain / target):
#   Hoechst 33342 (DNA), concanavalin A (ER), SYTO 14 (RNA),
#   phalloidin (actin/AGP), WGA (Golgi/plasma membrane, Mito merged)
CELL_PAINTING_CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_and_mask(
    n_channels: int = 2,
    size: int = 64,
    n_objects: int = 2,
    seed: int = 42,
    dtype=np.float64,
):
    """Create a small random image and labeled mask for testing.

    Returns image ``(C, H, W)`` and mask ``(1, H, W)`` (single mask plane).
    """
    rng = np.random.default_rng(seed)
    image = rng.random((n_channels, size, size)).astype(dtype)
    mask = np.zeros((1, size, size), dtype=np.int32)
    # Place non-overlapping square objects
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


class TestSmokeTest:
    def test_intensity_and_sizeshape(self):
        """Basic end-to-end: 2 channels, intensity + sizeshape."""
        channels = CELL_PAINTING_CHANNELS[:2]
        image, mask = _make_image_and_mask(n_channels=2)

        f = make_featurizer(channels, intensity=True, sizeshape=True)
        df = f.featurize(image, mask)

        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "label"
        assert len(df) == 2  # 2 objects
        # Should have intensity columns per channel and shape columns
        assert any("_DNA" in c for c in df.columns)
        assert any("_ER" in c for c in df.columns)

    def test_single_feature_works(self):
        """Enabling only one feature type should work."""
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], intensity=True)
        df = f.featurize(image, mask)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------


class TestColumnNaming:
    def test_per_channel_suffix(self):
        """Per-channel features have _{channel} suffix."""
        channels = CELL_PAINTING_CHANNELS[:2]
        image, mask = _make_image_and_mask(n_channels=2)

        f = make_featurizer(channels, intensity=True)
        df = f.featurize(image, mask)

        for col in df.columns:
            assert col.endswith("_DNA") or col.endswith("_ER"), (
                f"Expected channel suffix, got column: {col}"
            )

    def test_shape_features_no_channel_suffix(self):
        """Shape features have no channel suffix."""
        channels = CELL_PAINTING_CHANNELS[:2]
        image, mask = _make_image_and_mask(n_channels=2)

        f = make_featurizer(channels, sizeshape=True)
        df = f.featurize(image, mask)

        for col in df.columns:
            assert not col.endswith("_DNA") and not col.endswith("_ER"), (
                f"Shape feature should not have channel suffix: {col}"
            )

    def test_correlation_naming_asymmetric(self):
        """Asymmetric correlation features use permutations (both orderings)."""
        channels = CELL_PAINTING_CHANNELS[:3]
        image, mask = _make_image_and_mask(n_channels=3)

        # Pearson is asymmetric (Slope differs by ordering)
        f = make_featurizer(channels, correlation_pearson=True)
        df = f.featurize(image, mask)

        # Should have permutations: (A,B), (B,A) for all pairs
        expected_perms = list(itertools.permutations(channels, 2))
        for a, b in expected_perms:
            matching = [c for c in df.columns if c.endswith(f"_{a}_{b}")]
            assert len(matching) > 0, f"Missing permutation ({a}, {b})"

    def test_correlation_naming_symmetric(self):
        """Symmetric correlation features use combinations (one ordering)."""
        channels = CELL_PAINTING_CHANNELS[:3]
        image, mask = _make_image_and_mask(n_channels=3)

        # Manders is symmetric (_1/_2 capture both directions)
        f = make_featurizer(channels, correlation_manders_fold=True)
        df = f.featurize(image, mask)

        expected_combos = list(itertools.combinations(channels, 2))
        for col in df.columns:
            matched = any(col.endswith(f"_{a}_{b}") for a, b in expected_combos)
            assert matched, f"Column {col} should use combinations, not permutations"

    def test_multi_mask_column_prefixing(self):
        """All columns start with their respective mask name."""
        channels = CELL_PAINTING_CHANNELS[:2]
        image, mask = _make_image_and_mask(n_channels=2)

        mask_names = ["nuclei", "cells"]
        masks = np.concatenate([mask, mask], axis=0)  # (2, H, W)

        f = make_featurizer(channels, masks=mask_names, intensity=True, sizeshape=True)
        df = f.featurize(image, masks)

        for col in df.columns:
            assert col.startswith("nuclei_") or col.startswith("cells_"), (
                f"Column {col} does not start with a mask name"
            )


# ---------------------------------------------------------------------------
# Parameter forwarding
# ---------------------------------------------------------------------------


class TestParamsForwarding:
    def test_granularity_spectrum_length(self):
        """granularity_params are forwarded to the underlying function."""
        channels = ["DNA"]
        image, mask = _make_image_and_mask(n_channels=1)

        f4 = make_featurizer(
            channels,
            granularity=True,
            granularity_params={"granular_spectrum_length": 4},
        )
        df4 = f4.featurize(image, mask)
        granularity_cols_4 = [c for c in df4.columns if "Granularity" in c]

        f8 = make_featurizer(
            channels,
            granularity=True,
            granularity_params={"granular_spectrum_length": 8},
        )
        df8 = f8.featurize(image, mask)
        granularity_cols_8 = [c for c in df8.columns if "Granularity" in c]

        assert len(granularity_cols_8) > len(granularity_cols_4), (
            f"Expected more columns with spectrum_length=8 ({len(granularity_cols_8)}) "
            f"than 4 ({len(granularity_cols_4)})"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_image_not_3d(self):
        image = np.ones((10, 10))
        mask = np.ones((1, 10, 10), dtype=np.int32)
        f = make_featurizer(["DNA"], intensity=True)
        with pytest.raises(ValueError, match="3D"):
            f.featurize(image, mask)

    def test_masks_not_3d(self):
        image = np.ones((1, 10, 10))
        mask = np.ones((10, 10), dtype=np.int32)
        f = make_featurizer(["DNA"], intensity=True)
        with pytest.raises(ValueError, match="3D"):
            f.featurize(image, mask)

    def test_channel_count_mismatch(self):
        image = np.ones((3, 10, 10))
        mask = np.ones((1, 10, 10), dtype=np.int32)
        f = make_featurizer(["DNA", "ER"], intensity=True)
        with pytest.raises(ValueError, match="channels"):
            f.featurize(image, mask)

    def test_mask_count_mismatch(self):
        image = np.ones((1, 10, 10))
        masks = np.ones((2, 10, 10), dtype=np.int32)
        f = make_featurizer(["DNA"], masks=["nuclei"], intensity=True)
        with pytest.raises(ValueError, match="mask names"):
            f.featurize(image, masks)

    def test_spatial_dims_mismatch(self):
        image = np.ones((1, 10, 10))
        mask = np.ones((1, 8, 8), dtype=np.int32)
        f = make_featurizer(["DNA"], intensity=True)
        with pytest.raises(ValueError, match="spatial dims"):
            f.featurize(image, mask)

    def test_mask_not_integer(self):
        image = np.ones((1, 10, 10))
        mask = np.ones((1, 10, 10), dtype=np.float64)
        f = make_featurizer(["DNA"], intensity=True)
        with pytest.raises(TypeError, match="integer dtype"):
            f.featurize(image, mask)

    def test_all_masks_empty(self):
        image = np.ones((1, 10, 10))
        mask = np.zeros((1, 10, 10), dtype=np.int32)
        f = make_featurizer(["DNA"], intensity=True)
        with pytest.raises(ValueError, match="no labels"):
            f.featurize(image, mask)

    def test_non_contiguous_labels(self):
        """Mask with labels [1, 3] (gap at 2) should raise ValueError."""
        image = np.ones((1, 10, 10))
        mask = np.zeros((1, 10, 10), dtype=np.int32)
        mask[0, 0:3, 0:3] = 1
        mask[0, 5:8, 5:8] = 3  # gap: no label 2
        f = make_featurizer(["DNA"], intensity=True)
        with pytest.raises(ValueError, match="non-contiguous"):
            f.featurize(image, mask)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_integer_image_warns(self):
        """Integer-dtype image should trigger a UserWarning."""
        image, mask = _make_image_and_mask(n_channels=1, dtype=np.float64)
        image_uint8 = (image * 255).astype(np.uint8)
        f = make_featurizer(["DNA"], intensity=True)
        with pytest.warns(UserWarning, match="integer dtype"):
            f.featurize(image_uint8, mask)


# ---------------------------------------------------------------------------
# Multi-mask tests
# ---------------------------------------------------------------------------


class TestMultiMask:
    def test_multi_mask_smoke(self):
        """Two masks with different label counts → union index and NaN fill."""
        rng = np.random.default_rng(42)
        size = 64
        image = rng.random((2, size, size))

        # Nuclei mask: labels 1, 2
        nuclei = np.zeros((size, size), dtype=np.int32)
        nuclei[5:15, 5:15] = 1
        nuclei[30:40, 30:40] = 2

        # Cells mask: labels 1, 2, 3 (cell 3 has no nucleus)
        cells = np.zeros((size, size), dtype=np.int32)
        cells[3:18, 3:18] = 1
        cells[28:45, 28:45] = 2
        cells[50:60, 50:60] = 3

        masks = np.stack([nuclei, cells], axis=0)  # (2, H, W)

        f = make_featurizer(
            CELL_PAINTING_CHANNELS[:2],
            masks=["nuclei", "cells"],
            intensity=True,
            sizeshape=True,
        )
        df = f.featurize(image, masks)

        # Union index: labels 1, 2, 3
        assert len(df) == 3
        assert set(df.index) == {1, 2, 3}

        # Label 3 exists only in cells → nuclei columns should be NaN
        nuclei_cols = [c for c in df.columns if c.startswith("nuclei_")]
        cells_cols = [c for c in df.columns if c.startswith("cells_")]
        assert len(nuclei_cols) > 0
        assert len(cells_cols) > 0

        # Label 3 should have NaN for all nuclei columns
        assert df.loc[3, nuclei_cols].isna().all()
        # Label 3 should have values for most cells columns
        # (NormalizedMoment columns are mathematically NaN for certain shapes)
        known_nan_patterns = {
            "NormalizedMoment_0_0",
            "NormalizedMoment_0_1",
            "NormalizedMoment_1_0",
        }
        cells_cols_without_known_nan = [
            c for c in cells_cols if not any(p in c for p in known_nan_patterns)
        ]
        assert df.loc[3, cells_cols_without_known_nan].notna().all()

    def test_single_mask_default_name(self):
        """When masks param is omitted, default name is 'mask'."""
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], intensity=True)
        df = f.featurize(image, mask)

        for col in df.columns:
            assert col.startswith("mask_"), (
                f"Expected 'mask_' prefix with default mask name, got: {col}"
            )

    def test_mask_plane_with_no_labels_skipped(self):
        """A mask plane with all zeros is skipped gracefully."""
        rng = np.random.default_rng(42)
        size = 64
        image = rng.random((1, size, size))

        # First mask has labels, second is empty
        mask1 = np.zeros((size, size), dtype=np.int32)
        mask1[5:15, 5:15] = 1
        mask2 = np.zeros((size, size), dtype=np.int32)

        masks = np.stack([mask1, mask2], axis=0)

        f = make_featurizer(["DNA"], masks=["nuclei", "cells"], intensity=True)
        df = f.featurize(image, masks)

        # Only nuclei columns should be present
        assert all(c.startswith("nuclei_") for c in df.columns)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Factory validation
# ---------------------------------------------------------------------------


class TestMakeFeaturizer:
    def test_no_features_raises(self):
        with pytest.raises(ValueError, match="at least one feature"):
            make_featurizer(["DNA"])

    def test_empty_channels_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_featurizer([], intensity=True)

    def test_empty_masks_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_featurizer(["DNA"], masks=[], intensity=True)

    def test_duplicate_mask_names_raises(self):
        with pytest.raises(ValueError, match="unique"):
            make_featurizer(["DNA"], masks=["cells", "cells"], intensity=True)

    def test_single_channel_correlation_warns(self):
        """Single channel + correlation should warn and skip correlation."""
        with pytest.warns(UserWarning, match="at least 2 channels"):
            f = make_featurizer(["DNA"], intensity=True, correlation_pearson=True)
        assert len(f._correlation_features) == 0


# ---------------------------------------------------------------------------
# End-to-end realistic CellProfiler-style test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Realistic end-to-end test with a typical CellProfiler-sized image."""

    @pytest.fixture()
    def cellprofiler_data(self):
        """Simulate a typical CellProfiler image: 5 channels, ~1024x1024,
        multiple cells with nuclei and cell masks."""
        rng = np.random.default_rng(42)
        size = 1024
        channels = list(CELL_PAINTING_CHANNELS)
        n_channels = len(channels)

        # Simulate float64 image normalized to [0, 1] (CellProfiler convention)
        image = rng.random((n_channels, size, size))

        # Create nuclei mask with 10 objects (smaller circles)
        nuclei = np.zeros((size, size), dtype=np.int32)
        # Create cells mask with 12 objects (larger circles, includes nuclei labels)
        cells = np.zeros((size, size), dtype=np.int32)

        centers = [
            (150, 200),
            (300, 400),
            (500, 100),
            (600, 600),
            (100, 800),
            (400, 700),
            (750, 300),
            (800, 800),
            (200, 550),
            (900, 150),
        ]
        yy, xx = np.ogrid[:size, :size]

        # Nuclei: 10 objects with smaller radii
        for label, (cy, cx) in enumerate(centers, start=1):
            radius = rng.integers(15, 30)
            dist = (yy - cy) ** 2 + (xx - cx) ** 2
            nuclei[dist <= radius**2] = label

        # Cells: same 10 objects with larger radii + 2 extra cells without nuclei
        for label, (cy, cx) in enumerate(centers, start=1):
            radius = rng.integers(30, 55)
            dist = (yy - cy) ** 2 + (xx - cx) ** 2
            cells[dist <= radius**2] = label

        # Extra cells (labels 11, 12) without corresponding nuclei
        extra_centers = [(50, 50), (950, 950)]
        for i, (cy, cx) in enumerate(extra_centers, start=11):
            radius = rng.integers(20, 40)
            dist = (yy - cy) ** 2 + (xx - cx) ** 2
            cells[dist <= radius**2] = i

        masks = np.stack([nuclei, cells], axis=0)  # (2, H, W)
        mask_names = ["nuclei", "cells"]

        return image, masks, channels, mask_names

    def test_all_features(self, cellprofiler_data):
        """Run all feature categories on a realistic CellProfiler-sized image."""
        image, masks, channels, mask_names = cellprofiler_data
        nuclei_labels = masks[0].max()
        cells_labels = masks[1].max()

        f = make_featurizer(
            channels,
            masks=mask_names,
            # Per-channel intensity features
            intensity=True,
            texture=True,
            granularity=True,
            granularity_params={"granular_spectrum_length": 8},
            radial_distribution=True,
            radial_zernikes=True,
            # Shape features (mask only)
            sizeshape=True,
            ferret=True,
            # Correlation features (pairwise)
            correlation_pearson=True,
            correlation_manders_fold=True,
        )
        df = f.featurize(image, masks)

        # Basic shape checks
        assert isinstance(df, pd.DataFrame)
        # Union of labels: max(nuclei_labels, cells_labels)
        expected_labels = max(nuclei_labels, cells_labels)
        assert len(df) == expected_labels
        assert df.index.name == "label"
        np.testing.assert_array_equal(df.index, np.arange(1, expected_labels + 1))

        # Labels 11, 12 have no nuclei → nuclei columns should be NaN
        nuclei_cols = [c for c in df.columns if c.startswith("nuclei_")]
        for label in range(nuclei_labels + 1, cells_labels + 1):
            assert df.loc[label, nuclei_cols].isna().all(), (
                f"Label {label} should have NaN nuclei features"
            )

        # All columns should be prefixed with a mask name
        for col in df.columns:
            assert col.startswith("nuclei_") or col.startswith("cells_"), (
                f"Column {col} missing mask prefix"
            )

        # Per-channel columns exist for each channel under each mask
        for mask_name in mask_names:
            for ch in channels:
                matching = [
                    c
                    for c in df.columns
                    if c.startswith(f"{mask_name}_") and c.endswith(f"_{ch}")
                ]
                assert len(matching) > 0, (
                    f"No columns for mask={mask_name}, channel={ch}"
                )

        # All channel pair orderings (for filtering shape vs correlation cols)
        all_pairs = set(itertools.combinations(channels, 2)) | set(
            itertools.permutations(channels, 2)
        )

        # Shape columns exist (no channel suffix) under each mask
        for mask_name in mask_names:
            mask_cols = [c for c in df.columns if c.startswith(f"{mask_name}_")]
            shape_cols = [
                c
                for c in mask_cols
                if not any(c.endswith(f"_{ch}") for ch in channels)
                and not any(f"_{a}_{b}" in c for a, b in all_pairs)
            ]
            assert len(shape_cols) > 0, f"No shape columns for mask={mask_name}"

        # Correlation columns exist for channel pairs under each mask
        for mask_name in mask_names:
            corr_cols = [
                c
                for c in df.columns
                if c.startswith(f"{mask_name}_")
                and any(f"_{a}_{b}" in c for a, b in all_pairs)
            ]
            assert len(corr_cols) > 0, f"No correlation columns for mask={mask_name}"

        # Almost no fully-NaN columns (excluding labels that are in only one mask)
        # For labels 1-10 (present in both masks), check for unexpected NaN
        shared_df = df.loc[1:nuclei_labels]
        known_nan_patterns = {
            "NormalizedMoment_0_0",
            "NormalizedMoment_0_1",
            "NormalizedMoment_1_0",
        }
        all_nan_cols = set(shared_df.columns[shared_df.isna().all()].tolist())
        unexpected_nan = {
            c for c in all_nan_cols if not any(p in c for p in known_nan_patterns)
        }
        assert len(unexpected_nan) == 0, f"Unexpected all-NaN columns: {unexpected_nan}"

        # Reasonable column count: 2 masks × features
        assert len(df.columns) > 200, (
            f"Expected >200 total columns (2 masks), got {len(df.columns)}"
        )
