"""Tests for the high-level Featurizer orchestrator."""

import itertools
import warnings

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

# Column name patterns produced by 2D-only features; must be absent
# from volumetric output.
_2D_ONLY_PATTERNS = {"RadialDistribution", "RadialCV", "Zernike", "Feret"}

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

    Returns image ``(C, H, W)`` and mask ``(1, H, W)`` (single object mask).
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
        assert any("__DNA" in c for c in df.column_names)
        assert any("__ER" in c for c in df.column_names)

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
    def test_image_2d_raises(self):
        image = np.ones((10, 10))
        mask = np.ones((1, 10, 10), dtype=np.int32)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="3D.*4D"):
            f.featurize(image, mask)

    def test_image_5d_raises(self):
        image = np.ones((1, 2, 3, 4, 5))
        mask = np.ones((1, 2, 3, 4, 5), dtype=np.int32)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="3D.*4D"):
            f.featurize(image, mask)

    def test_masks_2d_raises(self):
        image = np.ones((1, 10, 10))
        mask = np.ones((10, 10), dtype=np.int32)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="3D.*4D"):
            f.featurize(image, mask)

    def test_ndim_mismatch(self):
        """3D image with 4D masks should raise ValueError."""
        image = np.ones((1, 10, 10))
        mask = np.ones((1, 2, 10, 10), dtype=np.int32)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="same number of dimensions"):
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
            ["DNA"], objects=["nuclei"], **{**_ALL_OFF, "intensity": True}
        )
        with pytest.raises(ValueError, match="object names"):
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
# Volumetric (3D spatial) tests
# ---------------------------------------------------------------------------


def _make_volumetric_image_and_mask(
    n_channels: int = 2,
    size: int = 32,
    depth: int = 8,
    n_objects: int = 2,
    seed: int = 42,
    dtype=np.float64,
):
    """Create a small random volumetric image and labeled mask.

    Returns image ``(C, Z, H, W)`` and mask ``(1, Z, H, W)`` (single object mask).
    """
    rng = np.random.default_rng(seed)
    image = rng.random((n_channels, depth, size, size)).astype(dtype)
    mask = np.zeros((1, depth, size, size), dtype=np.int32)
    # Place non-overlapping cubic objects
    step = size // (n_objects + 1)
    obj_size = max(step // 2, 4)
    z_size = min(obj_size, depth - 1)
    for i in range(n_objects):
        r = step * (i + 1) - obj_size // 2
        c = step * (i + 1) - obj_size // 2
        z = 1
        mask[0, z : z + z_size, r : r + obj_size, c : c + obj_size] = i + 1
    return image, mask


class TestVolumetric:
    def test_volumetric_smoke(self):
        """4D intensity + sizeshape runs and returns correct rows."""
        image, mask = _make_volumetric_image_and_mask(n_channels=2)
        f = make_featurizer(
            ["DNA", "ER"],
            **{**_ALL_OFF, "intensity": True, "sizeshape": True},
        )
        df = f.featurize(image, mask)

        assert isinstance(df, pa.Table)
        assert df.num_rows == 2
        assert "label" in df.column_names
        # Must produce actual feature columns beyond just "label"
        feat_cols = _feature_columns(df)
        assert len(feat_cols) > 0, "No feature columns produced"
        assert any("Intensity" in c for c in feat_cols)
        assert any("Area" in c or "Volume" in c for c in feat_cols)
        # No column should be entirely null
        all_null = [c for c in feat_cols if df.column(c).null_count == df.num_rows]
        assert len(all_null) == 0, f"All-null columns: {all_null}"

    def test_volumetric_skips_2d_only_features(self):
        """Volumetric input with all features warns and omits 2D-only columns."""
        image, mask = _make_volumetric_image_and_mask(n_channels=2)
        f = make_featurizer(["DNA", "ER"])
        with pytest.warns(UserWarning, match="2D-only"):
            df = f.featurize(image, mask)

        feat_cols = _feature_columns(df)
        # 2D-only feature column patterns must be absent
        present_2d = [c for c in feat_cols if any(p in c for p in _2D_ONLY_PATTERNS)]
        assert present_2d == [], f"2D-only columns in volumetric output: {present_2d}"

        # Volumetric features should be present
        assert any("Intensity" in c for c in feat_cols)
        assert any("Area" in c or "Volume" in c for c in feat_cols)

    def test_volumetric_only_2d_features_raises(self):
        """Volumetric input with only 2D-only features raises ValueError.

        The ValueError must fire *without* a preceding warning (the
        warning would be redundant when all features are filtered out).
        """
        image, mask = _make_volumetric_image_and_mask(n_channels=1)
        f = make_featurizer(
            ["DNA"],
            **{
                **_ALL_OFF,
                "radial_distribution": True,
                "zernike": True,
                "ferret": True,
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="no features left"):
                f.featurize(image, mask)

    def test_volumetric_correlation(self):
        """Correlation features work end-to-end on volumetric data."""
        image, mask = _make_volumetric_image_and_mask(n_channels=2)
        f = make_featurizer(
            ["DNA", "ER"],
            **{**_ALL_OFF, "intensity": True, "correlation_pearson": True},
        )
        df = f.featurize(image, mask)

        assert df.num_rows == 2
        corr_cols = [c for c in df.column_names if "Correlation" in c]
        assert len(corr_cols) > 0, "No correlation columns for volumetric input"


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
            objects=["nuclei", "cells"],
            **{**_ALL_OFF, "intensity": True, "sizeshape": True},
        )
        df = f.featurize(image, masks)

        # Union of labels: 1, 2, 3
        assert df.num_rows == 3
        assert set(df.column("label").to_pylist()) == {1, 2, 3}

        # Label 3 exists only in cells -> nuclei columns should be null
        nuclei_cols = [c for c in df.column_names if c.startswith("nuclei__")]
        cells_cols = [c for c in df.column_names if c.startswith("cells__")]
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
        """When objects param is omitted, default name is 'object'."""
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        df = f.featurize(image, mask)

        for col in _feature_columns(df):
            assert col.startswith("object__"), (
                f"Expected 'object__' prefix with default object name, got: {col}"
            )

    def test_empty_object_mask_skipped(self):
        """An object mask with all zeros is skipped gracefully."""
        rng = np.random.default_rng(42)
        size = 64
        image = rng.random((1, size, size))

        # First mask has labels, second is empty
        mask1 = np.zeros((size, size), dtype=np.int32)
        mask1[5:15, 5:15] = 1
        mask2 = np.zeros((size, size), dtype=np.int32)

        masks = np.stack([mask1, mask2], axis=0)

        f = make_featurizer(
            ["DNA"], objects=["nuclei", "cells"], **{**_ALL_OFF, "intensity": True}
        )
        df = f.featurize(image, masks)

        # Only nuclei columns (plus label) should be present
        assert all(c.startswith("nuclei__") for c in _feature_columns(df))
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

    def test_empty_objects_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_featurizer(["DNA"], objects=[], **{**_ALL_OFF, "intensity": True})

    def test_duplicate_object_names_raises(self):
        with pytest.raises(ValueError, match="unique"):
            make_featurizer(
                ["DNA"],
                objects=["cells", "cells"],
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

    def test_return_as_dict(self):
        """return_as='dict' returns a plain Python dict."""
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        result = f.featurize(image, mask, return_as="dict")
        assert isinstance(result, dict)
        assert "label" in result

    def test_return_as_invalid(self):
        """Invalid return_as raises ValueError."""
        image, mask = _make_image_and_mask(n_channels=1)
        f = make_featurizer(["DNA"], **{**_ALL_OFF, "intensity": True})
        with pytest.raises(ValueError, match="return_as"):
            f.featurize(image, mask, return_as="numpy")


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
        f = make_featurizer(channels, objects=mask_names)
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
        nuclei_cols = [c for c in df.column_names if c.startswith("nuclei__")]
        label_list = df.column("label").to_pylist()
        for label in range(nuclei_labels + 1, cells_labels + 1):
            row_idx = label_list.index(label)
            for col in nuclei_cols:
                assert df.column(col)[row_idx].as_py() is None, (
                    f"Label {label} should have null nuclei features for {col}"
                )

        # All feature columns should be prefixed with a mask name
        for col in _feature_columns(df):
            assert col.startswith("nuclei__") or col.startswith("cells__"), (
                f"Column {col} missing object prefix"
            )

        # Per-channel columns exist for each channel under each mask
        for obj_name in mask_names:
            for ch in channels:
                matching = [
                    c
                    for c in df.column_names
                    if c.startswith(f"{obj_name}__") and c.endswith(f"__{ch}")
                ]
                assert len(matching) > 0, (
                    f"No columns for object={obj_name}, channel={ch}"
                )

        # All channel pair orderings (for filtering shape vs correlation cols)
        all_pairs = set(itertools.combinations(channels, 2)) | set(
            itertools.permutations(channels, 2)
        )

        # Shape columns exist (no channel suffix) under each mask
        for obj_name in mask_names:
            obj_cols = [c for c in df.column_names if c.startswith(f"{obj_name}__")]
            shape_cols = [
                c
                for c in obj_cols
                if not any(c.endswith(f"__{ch}") for ch in channels)
                and not any(f"__{a}__{b}" in c for a, b in all_pairs)
            ]
            assert len(shape_cols) > 0, f"No shape columns for object={obj_name}"

        # Correlation columns exist for channel pairs under each mask
        for obj_name in mask_names:
            corr_cols = [
                c
                for c in df.column_names
                if c.startswith(f"{obj_name}__")
                and any(f"__{a}__{b}" in c for a, b in all_pairs)
            ]
            assert len(corr_cols) > 0, f"No correlation columns for object={obj_name}"

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

        # Smoke check: should have a substantial number of feature columns
        assert df.num_columns > 100, (
            f"Expected many feature columns, got {df.num_columns}"
        )
