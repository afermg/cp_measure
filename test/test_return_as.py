"""Tests for the return_as parameter of featurize()."""

import numpy as np
import pytest

from cp_measure.featurizer import featurize, make_featurizer_config

from conftest import ALL_OFF


@pytest.fixture()
def config_2ch():
    return make_featurizer_config(
        ["DNA", "ER"],
        objects=["nuclei"],
        **{
            **ALL_OFF,
            "intensity": True,
            "sizeshape": True,
            "correlation_pearson": True,
        },
    )


@pytest.fixture()
def config_2ch_multi():
    return make_featurizer_config(
        ["DNA", "ER"],
        objects=["nuclei", "cells"],
        **{
            **ALL_OFF,
            "intensity": True,
            "sizeshape": True,
            "correlation_pearson": True,
        },
    )


class TestReturnAsValidation:
    def test_invalid_return_as(self, image_2d_2ch, mask_2d, config_2ch):
        with pytest.raises(ValueError, match="return_as must be one of"):
            featurize(image_2d_2ch, mask_2d, config_2ch, return_as="invalid")

    def test_tuple_default(self, image_2d_2ch, mask_2d, config_2ch):
        result = featurize(image_2d_2ch, mask_2d, config_2ch)
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestReturnAsPandas:
    pd = pytest.importorskip("pandas")

    def test_returns_dataframe(self, image_2d_2ch, mask_2d, config_2ch):
        df = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pandas")
        assert isinstance(df, self.pd.DataFrame)

    def test_metadata_columns(self, image_2d_2ch, mask_2d, config_2ch):
        df = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pandas")
        assert "image_id" in df.columns
        assert "object_type" in df.columns
        assert "label" in df.columns
        assert df.columns[0] == "image_id"
        assert df.columns[1] == "object_type"
        assert df.columns[2] == "label"

    def test_row_count(self, image_2d_2ch, mask_2d, config_2ch):
        data, _, rows = featurize(image_2d_2ch, mask_2d, config_2ch)
        df = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pandas")
        assert len(df) == len(rows)
        assert len(df) == data.shape[0]

    def test_feature_columns_match(self, image_2d_2ch, mask_2d, config_2ch):
        _, columns, _ = featurize(image_2d_2ch, mask_2d, config_2ch)
        df = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pandas")
        feature_cols = df.columns[3:]
        assert list(feature_cols) == columns

    def test_values_match_tuple(self, image_2d_2ch, mask_2d, config_2ch):
        data, columns, _ = featurize(image_2d_2ch, mask_2d, config_2ch)
        df = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pandas")
        np.testing.assert_array_equal(df[columns].values, data)

    def test_object_type_values(self, image_2d_2ch, mask_2d, config_2ch):
        df = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pandas")
        assert (df["object_type"] == "nuclei").all()

    def test_multi_mask(self, image_2d_2ch, masks_2d_multi, config_2ch_multi):
        df = featurize(
            image_2d_2ch, masks_2d_multi, config_2ch_multi, return_as="pandas"
        )
        assert set(df["object_type"]) == {"nuclei", "cells"}
        assert len(df[df["object_type"] == "nuclei"]) == 2
        assert len(df[df["object_type"] == "cells"]) == 3

    def test_image_id(self, image_2d_2ch, mask_2d, config_2ch):
        df = featurize(
            image_2d_2ch, mask_2d, config_2ch, image_id="plate1", return_as="pandas"
        )
        assert (df["image_id"] == "plate1").all()


class TestReturnAsPyArrow:
    pa = pytest.importorskip("pyarrow")

    def test_returns_table(self, image_2d_2ch, mask_2d, config_2ch):
        table = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pyarrow")
        assert isinstance(table, self.pa.Table)

    def test_metadata_columns(self, image_2d_2ch, mask_2d, config_2ch):
        table = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pyarrow")
        names = table.column_names
        assert names[0] == "image_id"
        assert names[1] == "object_type"
        assert names[2] == "label"

    def test_row_count(self, image_2d_2ch, mask_2d, config_2ch):
        data, _, _ = featurize(image_2d_2ch, mask_2d, config_2ch)
        table = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pyarrow")
        assert table.num_rows == data.shape[0]

    def test_schema_metadata(self, image_2d_2ch, mask_2d, config_2ch):
        table = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pyarrow")
        meta = table.schema.metadata
        assert b"cp_measure_config" in meta
        assert b"channels" in meta
        assert b"is_3d" in meta

    def test_column_metadata(self, image_2d_2ch, mask_2d, config_2ch):
        table = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pyarrow")
        # Feature columns (index >= 3) should have metadata
        field = table.schema.field(3)
        assert field.metadata is not None
        assert b"feature_group" in field.metadata

    def test_feature_columns_match(self, image_2d_2ch, mask_2d, config_2ch):
        _, columns, _ = featurize(image_2d_2ch, mask_2d, config_2ch)
        table = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="pyarrow")
        assert table.column_names[3:] == columns


class TestReturnAsAnnData:
    ad = pytest.importorskip("anndata")

    def test_returns_anndata(self, image_2d_2ch, mask_2d, config_2ch):
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        assert isinstance(adata, self.ad.AnnData)

    def test_x_shape(self, image_2d_2ch, mask_2d, config_2ch):
        data, columns, rows = featurize(image_2d_2ch, mask_2d, config_2ch)
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        assert adata.X.shape == data.shape
        assert adata.n_obs == len(rows)
        assert adata.n_vars == len(columns)

    def test_x_values_match(self, image_2d_2ch, mask_2d, config_2ch):
        data, _, _ = featurize(image_2d_2ch, mask_2d, config_2ch)
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        np.testing.assert_array_almost_equal(adata.X, data.astype(np.float32))

    def test_obs_columns(self, image_2d_2ch, mask_2d, config_2ch):
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        assert "image_id" in adata.obs.columns
        assert "object_type" in adata.obs.columns
        assert "label" in adata.obs.columns

    def test_obs_values(self, image_2d_2ch, mask_2d, config_2ch):
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        assert (adata.obs["object_type"] == "nuclei").all()
        assert list(adata.obs["label"]) == [1, 2]

    def test_obs_names_with_image_id(self, image_2d_2ch, mask_2d, config_2ch):
        adata = featurize(
            image_2d_2ch, mask_2d, config_2ch, image_id="img1", return_as="anndata"
        )
        assert adata.obs_names[0] == "img1_nuclei_1"

    def test_obs_names_without_image_id(self, image_2d_2ch, mask_2d, config_2ch):
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        assert adata.obs_names[0] == "nuclei_1"

    def test_var_columns(self, image_2d_2ch, mask_2d, config_2ch):
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        for col in (
            "feature_group",
            "feature_type",
            "feature_name",
            "channel",
            "channel_2",
        ):
            assert col in adata.var.columns

    def test_var_names(self, image_2d_2ch, mask_2d, config_2ch):
        _, columns, _ = featurize(image_2d_2ch, mask_2d, config_2ch)
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        assert list(adata.var_names) == columns

    def test_var_shape_features(self, image_2d_2ch, mask_2d, config_2ch):
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        shape_vars = adata.var[adata.var["feature_type"] == "shape"]
        assert len(shape_vars) > 0
        assert shape_vars["channel"].isna().all()
        assert (shape_vars["feature_group"] == "sizeshape").all()

    def test_var_channel_features(self, image_2d_2ch, mask_2d, config_2ch):
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        ch_vars = adata.var[adata.var["feature_type"] == "channel"]
        assert len(ch_vars) > 0
        assert set(ch_vars["channel"].dropna()) == {"DNA", "ER"}

    def test_var_correlation_features(self, image_2d_2ch, mask_2d, config_2ch):
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        corr_vars = adata.var[adata.var["feature_type"] == "correlation"]
        assert len(corr_vars) > 0
        assert corr_vars["channel"].notna().all()
        assert corr_vars["channel_2"].notna().all()

    def test_uns_keys(self, image_2d_2ch, mask_2d, config_2ch):
        adata = featurize(image_2d_2ch, mask_2d, config_2ch, return_as="anndata")
        assert "config" in adata.uns
        assert "channels" in adata.uns
        assert "objects" in adata.uns
        assert "is_3d" in adata.uns
        assert adata.uns["channels"] == ["DNA", "ER"]
        assert adata.uns["objects"] == ["nuclei"]
        assert adata.uns["is_3d"] is False

    def test_multi_mask(self, image_2d_2ch, masks_2d_multi, config_2ch_multi):
        adata = featurize(
            image_2d_2ch, masks_2d_multi, config_2ch_multi, return_as="anndata"
        )
        assert adata.n_obs == 5  # 2 nuclei + 3 cells
        assert set(adata.obs["object_type"]) == {"nuclei", "cells"}

    def test_3d_uns_flag(self, image_3d_2ch, mask_3d):
        config = make_featurizer_config(
            ["DNA", "ER"], **{**ALL_OFF, "intensity": True, "sizeshape": True}
        )
        adata = featurize(image_3d_2ch, mask_3d, config, return_as="anndata")
        assert adata.uns["is_3d"] is True
