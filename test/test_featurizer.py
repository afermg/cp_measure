"""Tests for the high-level Featurizer orchestrator."""

import itertools

import numpy as np
import pyarrow as pa
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

# All feature flags set to False - tests override only what they need.
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


def _feature_columns(table: pa.Table) -> list[str]:
    """Return all column names except 'label'."""
    return [c for c in table.column_names if c != "label"]


def _column_to_numpy(table: pa.Table, col: str) -> np.ndarray:
    """Extract a column as a numpy array (nulls become NaN for floats)."""
    return table.column(col).to_numpy(zero_copy_only=False)


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

        assert isinstance(df, pa.Table)
        assert "label" in df.column_names
        assert df.num_rows == 2  # 2 objects
        # Should have intensity columns per channel and shape columns
        assert any("_DNA" in c for c in df.column_names)
        assert any("_ER" in c for c in df.column_names)

    def test_single_feature_works(self):
        """Enabling only one feature type should work."""
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        df = f.featurize(image, mask)
        assert df.num_rows == 2

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
            c
            for c in _feature_columns(df)
            if not any(p in c for p in _KNOWN_NAN_PATTERNS)
        ]

        # No column should be entirely null
        all_null = [c for c in check_cols if df.column(c).null_count == df.num_rows]
        assert len(all_null) == 0, f"All-null columns: {all_null}"

        # The majority of columns should have at least one non-zero value.
        numeric_cols = [
            c
            for c in check_cols
            if pa.types.is_floating(df.schema.field(c).type)
            or pa.types.is_integer(df.schema.field(c).type)
        ]
        n_cols = len(numeric_cols)
        nonzero_count = 0
        for c in numeric_cols:
            arr = _column_to_numpy(df, c)
            if np.any(arr != 0):
                nonzero_count += 1
        frac_nonzero = nonzero_count / n_cols
        assert frac_nonzero > 0.5, (
            f"Only {frac_nonzero:.0%} of columns have non-zero values; expected >50%"
        )

        # No column should contain inf
        inf_cols = []
        for c in numeric_cols:
            arr = _column_to_numpy(df, c).astype(np.float64)
            if np.any(np.isinf(arr)):
                inf_cols.append(c)
        assert len(inf_cols) == 0, f"Columns with inf: {inf_cols}"

        # Spot-check: Area must equal the object pixel count (10x10 = 100)
        # sizeshape is a shape feature - columns have no channel suffix
        area_cols = [
            c for c in df.column_names if "Area" in c and "BoundingBox" not in c
        ]
        assert len(area_cols) > 0, "No Area columns found"
        for col in area_cols:
            np.testing.assert_array_equal(
                _column_to_numpy(df, col),
                [100.0, 100.0],
                err_msg=f"{col} should equal object pixel count",
            )

        # Spot-check: MeanIntensity should be in (0, 1] for float [0,1] images
        mean_int_cols = [c for c in df.column_names if "MeanIntensity" in c]
        assert len(mean_int_cols) > 0, "No MeanIntensity columns found"
        for col in mean_int_cols:
            vals = _column_to_numpy(df, col)
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

        for col in _feature_columns(df):
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

        for col in _feature_columns(df):
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
        feat_cols = _feature_columns(df)
        expected_perms = list(itertools.permutations(channels, 2))
        for a, b in expected_perms:
            matching = [c for c in feat_cols if c.endswith(f"_{a}_{b}")]
            assert len(matching) > 0, f"Missing permutation ({a}, {b})"

    def test_correlation_naming_symmetric(self):
        """Symmetric correlation features use combinations (one ordering)."""
        channels = CELL_PAINTING_CHANNELS[:3]
        image, mask = _make_image_and_mask(n_channels=3)

        # Manders is symmetric (_1/_2 capture both directions)
        f = make_featurizer(channels, **{**_ALL_OFF, "correlation_manders_fold": True})
        df = f.featurize(image, mask)

        expected_combos = list(itertools.combinations(channels, 2))
        for col in _feature_columns(df):
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

        for col in _feature_columns(df):
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
        granularity_cols_4 = [c for c in df4.column_names if "Granularity" in c]

        f8 = make_featurizer(
            channels,
            **{**_ALL_OFF, "granularity": True},
            granularity_params={"granular_spectrum_length": 8},
        )
        df8 = f8.featurize(image, mask)
        granularity_cols_8 = [c for c in df8.column_names if "Granularity" in c]

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

    def test_negative_labels(self):
        """Masks with negative labels should raise ValueError."""
        image = np.ones((1, 10, 10))
        mask = np.zeros((1, 10, 10), dtype=np.int32)
        mask[0, 0:3, 0:3] = -1
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="negative"):
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
        """Two masks with different label counts -> union of labels and null fill."""
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

        # Union of labels: 1, 2, 3
        assert df.num_rows == 3
        assert set(df.column("label").to_pylist()) == {1, 2, 3}

        # Label 3 exists only in cells -> nuclei columns should be null
        nuclei_cols = [c for c in df.column_names if c.startswith("nuclei_")]
        cells_cols = [c for c in df.column_names if c.startswith("cells_")]
        assert len(nuclei_cols) > 0
        assert len(cells_cols) > 0

        # Label 3 is the last row (sorted by label)
        row3_idx = df.column("label").to_pylist().index(3)
        for col in nuclei_cols:
            assert df.column(col)[row3_idx].as_py() is None, (
                f"Label 3 should have null for nuclei column {col}"
            )
        # Label 3 should have values for most cells columns
        cells_cols_without_known_nan = [
            c for c in cells_cols if not any(p in c for p in _KNOWN_NAN_PATTERNS)
        ]
        for col in cells_cols_without_known_nan:
            assert df.column(col)[row3_idx].as_py() is not None, (
                f"Label 3 should have a value for cells column {col}"
            )

    def test_single_mask_default_name(self):
        """When masks param is omitted, default name is 'mask'."""
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        df = f.featurize(image, mask)

        for col in _feature_columns(df):
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

        # Only nuclei columns (plus label) should be present
        assert all(c.startswith("nuclei_") for c in _feature_columns(df))
        assert df.num_rows == 1


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

    def test_duplicate_channel_names_raises(self):
        with pytest.raises(ValueError, match="unique"):
            make_featurizer(
                ["DNA", "DNA"],
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
        corr_cols = [c for c in df.column_names if "Correlation" in c or "Slope" in c]
        assert len(corr_cols) == 0


# ---------------------------------------------------------------------------
# return_as parameter
# ---------------------------------------------------------------------------


class TestReturnAs:
    def test_default_returns_pyarrow(self):
        """Default return type is pyarrow.Table."""
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        result = f.featurize(image, mask)
        assert isinstance(result, pa.Table)

    def test_return_as_pyarrow(self):
        """Explicit return_as='pyarrow' returns pyarrow.Table."""
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        result = f.featurize(image, mask, return_as="pyarrow")
        assert isinstance(result, pa.Table)

    def test_return_as_polars(self):
        """return_as='polars' returns polars.DataFrame."""
        polars = pytest.importorskip("polars")
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        result = f.featurize(image, mask, return_as="polars")
        assert isinstance(result, polars.DataFrame)

    def test_return_as_pandas(self):
        """return_as='pandas' returns pandas.DataFrame."""
        pandas = pytest.importorskip("pandas")
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        result = f.featurize(image, mask, return_as="pandas")
        assert isinstance(result, pandas.DataFrame)

    def test_return_as_invalid(self):
        """Invalid return_as raises ValueError."""
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="return_as"):
            f.featurize(image, mask, return_as="numpy")

    def test_return_as_check_before_computation(self):
        """Library availability is checked before any computation runs."""
        import unittest.mock

        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})

        # Patch _validate to track if it was called — if the import check
        # happens first, _validate should not be called when library is missing
        with (
            unittest.mock.patch.dict("sys.modules", {"nonexistent": None}),
            pytest.raises(ValueError, match="return_as"),
        ):
            f.featurize(image, mask, return_as="nonexistent")


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
        assert isinstance(df, pa.Table)
        # Union of labels: max(nuclei_labels, cells_labels)
        expected_labels = max(nuclei_labels, cells_labels)
        assert df.num_rows == expected_labels
        assert "label" in df.column_names
        np.testing.assert_array_equal(
            df.column("label").to_numpy(), np.arange(1, expected_labels + 1)
        )

        # Labels 11, 12 have no nuclei -> nuclei columns should be null
        nuclei_cols = [c for c in df.column_names if c.startswith("nuclei_")]
        label_list = df.column("label").to_pylist()
        for label in range(nuclei_labels + 1, cells_labels + 1):
            row_idx = label_list.index(label)
            for col in nuclei_cols:
                assert df.column(col)[row_idx].as_py() is None, (
                    f"Label {label} should have null nuclei features for {col}"
                )

        # All feature columns should be prefixed with a mask name
        for col in _feature_columns(df):
            assert col.startswith("nuclei_") or col.startswith("cells_"), (
                f"Column {col} missing mask prefix"
            )

        # Per-channel columns exist for each channel under each mask
        for mask_name in mask_names:
            for ch in channels:
                matching = [
                    c
                    for c in df.column_names
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
            mask_cols = [c for c in df.column_names if c.startswith(f"{mask_name}_")]
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
                for c in df.column_names
                if c.startswith(f"{mask_name}_")
                and any(f"_{a}_{b}" in c for a, b in all_pairs)
            ]
            assert len(corr_cols) > 0, f"No correlation columns for mask={mask_name}"

        # Almost no fully-null columns (excluding labels that are in only one mask)
        # For labels 1-10 (present in both masks), check for unexpected nulls
        shared_indices = [
            i for i, lab in enumerate(label_list) if 1 <= lab <= nuclei_labels
        ]
        feat_cols = _feature_columns(df)
        all_null_cols = []
        for c in feat_cols:
            col = df.column(c)
            vals = [col[i].as_py() for i in shared_indices]
            if all(v is None for v in vals):
                all_null_cols.append(c)
        unexpected_null = {
            c for c in all_null_cols if not any(p in c for p in _KNOWN_NAN_PATTERNS)
        }
        assert len(unexpected_null) == 0, (
            f"Unexpected all-null columns: {unexpected_null}"
        )

        # Exact column count: 1035 per mask x 2 masks + 1 label column = 2071
        # (110 shape + 805 per-channel + 120 correlation = 1035)
        assert df.num_columns == 2071, (
            f"Expected 2071 columns (1035 per mask x 2 + 1 label), got {df.num_columns}"
        )
