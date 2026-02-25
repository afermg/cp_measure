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

# Shape features that are mathematically NaN for certain object geometries
# (e.g., uniform pixel values produce degenerate normalized moments).
_KNOWN_NAN_PATTERNS = {
    "NormalizedMoment_0_0",
    "NormalizedMoment_0_1",
    "NormalizedMoment_1_0",
}

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

        f = make_featurizer(
            channels, **{**_ALL_OFF, "intensity": True, "sizeshape": True}
        )
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
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        df = f.featurize(image, mask)
        assert len(df) == 2

    def test_values_are_finite_and_nontrivial(self):
        """Verify that features produce finite, non-trivial values."""
        channels = CELL_PAINTING_CHANNELS[:2]
        image, mask = _make_image_and_mask(n_channels=2)

        f = make_featurizer(
            channels, **{**_ALL_OFF, "intensity": True, "sizeshape": True}
        )
        df = f.featurize(image, mask)

        # Exclude columns known to produce NaN for certain geometries
        check_cols = [
            c for c in df.columns if not any(p in c for p in _KNOWN_NAN_PATTERNS)
        ]
        subset = df[check_cols]

        # No column should be entirely NaN
        all_nan = subset.columns[subset.isna().all()].tolist()
        assert len(all_nan) == 0, f"All-NaN columns: {all_nan}"

        # The majority of columns should have at least one non-zero value.
        # Some features are legitimately zero for symmetric objects (e.g.,
        # Eccentricity=0 for squares, odd-order CentralMoments for symmetric
        # shapes, Z coordinates for 2D images).
        numeric = subset.select_dtypes(include=[np.number])
        nonzero_cols = (numeric != 0).any()
        frac_nonzero = nonzero_cols.sum() / len(nonzero_cols)
        assert frac_nonzero > 0.5, (
            f"Only {frac_nonzero:.0%} of columns have non-zero values; expected >50%"
        )

        # No column should contain inf
        inf_cols = numeric.columns[np.isinf(numeric).any()].tolist()
        assert len(inf_cols) == 0, f"Columns with inf: {inf_cols}"

        # Spot-check: Area must equal the object pixel count (10x10 = 100)
        # sizeshape is a shape feature — columns have no channel suffix
        area_cols = [c for c in df.columns if "Area" in c and "BoundingBox" not in c]
        assert len(area_cols) > 0, "No Area columns found"
        for col in area_cols:
            np.testing.assert_array_equal(
                df[col].values,
                [100.0, 100.0],
                err_msg=f"{col} should equal object pixel count",
            )

        # Spot-check: MeanIntensity should be in (0, 1] for float [0,1] images
        mean_int_cols = [c for c in df.columns if "MeanIntensity" in c]
        assert len(mean_int_cols) > 0, "No MeanIntensity columns found"
        for col in mean_int_cols:
            vals = df[col].values
            assert np.all(vals > 0) and np.all(vals <= 1), (
                f"{col} values {vals} not in (0, 1]"
            )


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------


class TestColumnNaming:
    def test_per_channel_suffix(self):
        """Per-channel features have _{channel} suffix."""
        channels = CELL_PAINTING_CHANNELS[:2]
        image, mask = _make_image_and_mask(n_channels=2)

        f = make_featurizer(channels, **{**_ALL_OFF, "intensity": True})
        df = f.featurize(image, mask)

        for col in df.columns:
            assert col.endswith("_DNA") or col.endswith("_ER"), (
                f"Expected channel suffix, got column: {col}"
            )

    def test_shape_features_no_channel_suffix(self):
        """Purely geometric shape features (sizeshape, zernike, ferret) have no channel suffix."""
        channels = CELL_PAINTING_CHANNELS[:2]
        image, mask = _make_image_and_mask(n_channels=2)

        f = make_featurizer(
            channels,
            **{**_ALL_OFF, "sizeshape": True, "zernike": True, "ferret": True},
        )
        df = f.featurize(image, mask)

        for col in df.columns:
            assert not col.endswith("_DNA") and not col.endswith("_ER"), (
                f"Geometric shape feature should not have channel suffix: {col}"
            )

    def test_correlation_naming_asymmetric(self):
        """Asymmetric correlation features use permutations (both orderings)."""
        channels = CELL_PAINTING_CHANNELS[:3]
        image, mask = _make_image_and_mask(n_channels=3)

        # Pearson is asymmetric (Slope differs by ordering)
        f = make_featurizer(channels, **{**_ALL_OFF, "correlation_pearson": True})
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
        f = make_featurizer(channels, **{**_ALL_OFF, "correlation_manders_fold": True})
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

        f = make_featurizer(
            channels,
            masks=mask_names,
            **{**_ALL_OFF, "intensity": True, "sizeshape": True},
        )
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
            **{**_ALL_OFF, "granularity": True},
            granularity_params={"granular_spectrum_length": 4},
        )
        df4 = f4.featurize(image, mask)
        granularity_cols_4 = [c for c in df4.columns if "Granularity" in c]

        f8 = make_featurizer(
            channels,
            **{**_ALL_OFF, "granularity": True},
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
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="3D"):
            f.featurize(image, mask)

    def test_masks_not_3d(self):
        image = np.ones((1, 10, 10))
        mask = np.ones((10, 10), dtype=np.int32)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="3D"):
            f.featurize(image, mask)

    def test_channel_count_mismatch(self):
        image = np.ones((3, 10, 10))
        mask = np.ones((1, 10, 10), dtype=np.int32)
        f = make_featurizer(["DNA", "ER"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="channels"):
            f.featurize(image, mask)

    def test_mask_count_mismatch(self):
        image = np.ones((1, 10, 10))
        masks = np.ones((2, 10, 10), dtype=np.int32)
        f = make_featurizer(
            ["DNA"], masks=["nuclei"], **{**_ALL_OFF, "intensity": True}
        )
        with pytest.raises(ValueError, match="mask names"):
            f.featurize(image, masks)

    def test_spatial_dims_mismatch(self):
        image = np.ones((1, 10, 10))
        mask = np.ones((1, 8, 8), dtype=np.int32)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="spatial dims"):
            f.featurize(image, mask)

    def test_mask_not_integer(self):
        image = np.ones((1, 10, 10))
        mask = np.ones((1, 10, 10), dtype=np.float64)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(TypeError, match="integer dtype"):
            f.featurize(image, mask)

    def test_all_masks_empty(self):
        image = np.ones((1, 10, 10))
        mask = np.zeros((1, 10, 10), dtype=np.int32)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="no labels"):
            f.featurize(image, mask)

    def test_non_contiguous_labels(self):
        """Mask with labels [1, 3] (gap at 2) should raise ValueError."""
        image = np.ones((1, 10, 10))
        mask = np.zeros((1, 10, 10), dtype=np.int32)
        mask[0, 0:3, 0:3] = 1
        mask[0, 5:8, 5:8] = 3  # gap: no label 2
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
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
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
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
            **{**_ALL_OFF, "intensity": True, "sizeshape": True},
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
        # (some moment columns are mathematically NaN for uniform pixels)
        cells_cols_without_known_nan = [
            c for c in cells_cols if not any(p in c for p in _KNOWN_NAN_PATTERNS)
        ]
        assert df.loc[3, cells_cols_without_known_nan].notna().all()

    def test_single_mask_default_name(self):
        """When masks param is omitted, default name is 'mask'."""
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
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

        f = make_featurizer(
            ["DNA"], masks=["nuclei", "cells"], **{**_ALL_OFF, "intensity": True}
        )
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
            make_featurizer(["DNA"], **_ALL_OFF)

    def test_empty_channels_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_featurizer([], **{**_ALL_OFF, "intensity": True})

    def test_empty_masks_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_featurizer(["DNA"], masks=[], **{**_ALL_OFF, "intensity": True})

    def test_duplicate_mask_names_raises(self):
        with pytest.raises(ValueError, match="unique"):
            make_featurizer(
                ["DNA"],
                masks=["cells", "cells"],
                **{**_ALL_OFF, "intensity": True},
            )

    def test_single_channel_correlation_warns(self):
        """Single channel + correlation should warn and produce no correlation columns."""
        with pytest.warns(UserWarning, match="at least 2 channels"):
            f = make_featurizer(
                ["DNA"],
                **{**_ALL_OFF, "intensity": True, "correlation_pearson": True},
            )
        image, mask = _make_image_and_mask(n_channels=1)
        df = f.featurize(image, mask)
        corr_cols = [c for c in df.columns if "Correlation" in c or "Slope" in c]
        assert len(corr_cols) == 0


# ---------------------------------------------------------------------------
# End-to-end realistic CellProfiler-style test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Realistic end-to-end test with a typical CellProfiler-sized image."""

    @pytest.fixture()
    def cellprofiler_data(self):
        """Simulate a CellProfiler image: 5 channels, 512x512,
        multiple cells with nuclei and cell masks."""
        rng = np.random.default_rng(42)
        size = 512
        channels = CELL_PAINTING_CHANNELS
        n_channels = len(channels)

        image = rng.random((n_channels, size, size))

        nuclei = np.zeros((size, size), dtype=np.int32)
        cells = np.zeros((size, size), dtype=np.int32)

        centers = [
            (80, 100),
            (160, 200),
            (260, 60),
            (300, 300),
            (60, 400),
            (200, 350),
            (380, 150),
            (400, 400),
            (100, 280),
            (460, 80),
        ]
        yy, xx = np.ogrid[:size, :size]

        for label, (cy, cx) in enumerate(centers, start=1):
            radius = rng.integers(12, 24)
            dist = (yy - cy) ** 2 + (xx - cx) ** 2
            nuclei[dist <= radius**2] = label

        for label, (cy, cx) in enumerate(centers, start=1):
            radius = rng.integers(25, 45)
            dist = (yy - cy) ** 2 + (xx - cx) ** 2
            cells[dist <= radius**2] = label

        # Extra cells (labels 11, 12) without corresponding nuclei
        extra_centers = [(30, 30), (480, 480)]
        for i, (cy, cx) in enumerate(extra_centers, start=11):
            radius = rng.integers(12, 28)
            dist = (yy - cy) ** 2 + (xx - cx) ** 2
            cells[dist <= radius**2] = i

        masks = np.stack([nuclei, cells], axis=0)
        mask_names = ["nuclei", "cells"]

        return image, masks, channels, mask_names

    def test_all_features(self, cellprofiler_data):
        """Run ALL feature categories on a realistic CellProfiler-sized image."""
        image, masks, channels, mask_names = cellprofiler_data
        nuclei_labels = masks[0].max()
        cells_labels = masks[1].max()

        # All features enabled by default
        f = make_featurizer(channels, masks=mask_names)
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
        all_nan_cols = set(shared_df.columns[shared_df.isna().all()].tolist())
        unexpected_nan = {
            c for c in all_nan_cols if not any(p in c for p in _KNOWN_NAN_PATTERNS)
        }
        assert len(unexpected_nan) == 0, f"Unexpected all-NaN columns: {unexpected_nan}"

        # Exact column count: 1035 per mask × 2 masks
        # (110 shape + 805 per-channel + 120 correlation = 1035)
        assert len(df.columns) == 2070, (
            f"Expected 2070 columns (1035 per mask × 2), got {len(df.columns)}"
        )
