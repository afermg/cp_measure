"""High-level featurization wrapper for cp_measure.

Provides two stateless functions:

- :func:`make_featurizer_config` builds a plain configuration dictionary.
- :func:`featurize` takes that configuration together with image and mask
  arrays and returns a numpy feature matrix with column and row metadata.

Example
-------
>>> from cp_measure.featurizer import make_featurizer_config, featurize
>>> config = make_featurizer_config(["DNA", "ER"], objects=["nuclei", "cells"])
>>> data, columns, rows = featurize(image, masks, config)
"""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd
    import pyarrow as pa

# Feature groups that only support 2D spatial data.
_2D_ONLY = {"radial_distribution", "radial_zernikes", "zernike", "feret"}


def make_featurizer_config(
    channels: list[str] | None = None,
    *,
    objects: list[str] | None = None,
    intensity: bool = True,
    intensity_params: dict | None = None,
    texture: bool = True,
    texture_params: dict | None = None,
    granularity: bool = True,
    granularity_params: dict | None = None,
    radial_distribution: bool = True,
    radial_distribution_params: dict | None = None,
    radial_zernikes: bool = True,
    radial_zernikes_params: dict | None = None,
    sizeshape: bool = True,
    sizeshape_params: dict | None = None,
    zernike: bool = True,
    zernike_params: dict | None = None,
    feret: bool = True,
    correlation_pearson: bool = True,
    correlation_costes: bool = True,
    correlation_costes_params: dict | None = None,
    correlation_manders_fold: bool = True,
    correlation_manders_fold_params: dict | None = None,
    correlation_rwc: bool = True,
    correlation_rwc_params: dict | None = None,
) -> dict:
    """Build a featurizer configuration dictionary.

    The returned dictionary can be passed to :func:`featurize`.  It is
    plain data (no callables, no state) and can be serialised, compared,
    or used as a "published config" (e.g. JUMP parameters).

    Parameters
    ----------
    channels : list[str], optional
        Names for each channel in the image.  If ``None`` a warning is
        emitted and channels are auto-named ``ch0, ch1, …`` (zero-padded
        when there are 10 or more channels).
    objects : list[str], optional
        Names for each object mask.  Defaults to ``["object"]``.
    intensity, texture, granularity, radial_distribution, radial_zernikes,
    sizeshape, zernike, feret : bool
        Enable / disable individual feature groups.
    intensity_params, texture_params, granularity_params,
    radial_distribution_params, radial_zernikes_params, sizeshape_params,
    zernike_params : dict, optional
        Extra keyword arguments forwarded to the underlying functions.
    correlation_pearson, correlation_costes, correlation_manders_fold,
    correlation_rwc : bool
        Enable / disable correlation feature groups.
    correlation_costes_params, correlation_manders_fold_params,
    correlation_rwc_params : dict, optional
        Extra keyword arguments forwarded to the underlying functions.

    Returns
    -------
    dict
        Configuration dictionary accepted by :func:`featurize`.

    Raises
    ------
    ValueError
        If no features are enabled, if channel/object names are not
        unique, or if ``objects`` is explicitly empty.

    Examples
    --------
    >>> config = make_featurizer_config(["DNA", "ER"], objects=["nuclei", "cells"])
    >>> config = make_featurizer_config()  # auto-named channels, single "object" mask
    """
    if channels is not None:
        if len(set(channels)) != len(channels):
            raise ValueError("channel names must be unique")

    if objects is None:
        objects = ["object"]
    if not objects:
        raise ValueError("objects must be a non-empty list of object names")
    if len(set(objects)) != len(objects):
        raise ValueError("object names must be unique")

    _feature_flags = [
        intensity,
        texture,
        granularity,
        radial_distribution,
        radial_zernikes,
        sizeshape,
        zernike,
        feret,
        correlation_pearson,
        correlation_costes,
        correlation_manders_fold,
        correlation_rwc,
    ]
    if not any(_feature_flags):
        raise ValueError(
            "at least one feature must be enabled "
            "(e.g., intensity=True, sizeshape=True)"
        )

    return {
        "channels": list(channels) if channels is not None else None,
        "objects": list(objects),
        "intensity": intensity,
        "intensity_params": intensity_params if intensity_params is not None else {},
        "texture": texture,
        "texture_params": texture_params if texture_params is not None else {},
        "granularity": granularity,
        "granularity_params": granularity_params
        if granularity_params is not None
        else {},
        "radial_distribution": radial_distribution,
        "radial_distribution_params": radial_distribution_params
        if radial_distribution_params is not None
        else {},
        "radial_zernikes": radial_zernikes,
        "radial_zernikes_params": radial_zernikes_params
        if radial_zernikes_params is not None
        else {},
        "sizeshape": sizeshape,
        "sizeshape_params": sizeshape_params if sizeshape_params is not None else {},
        "zernike": zernike,
        "zernike_params": zernike_params if zernike_params is not None else {},
        "feret": feret,
        "correlation_pearson": correlation_pearson,
        "correlation_costes": correlation_costes,
        "correlation_costes_params": correlation_costes_params
        if correlation_costes_params is not None
        else {},
        "correlation_manders_fold": correlation_manders_fold,
        "correlation_manders_fold_params": correlation_manders_fold_params
        if correlation_manders_fold_params is not None
        else {},
        "correlation_rwc": correlation_rwc,
        "correlation_rwc_params": correlation_rwc_params
        if correlation_rwc_params is not None
        else {},
    }


@overload
def featurize(
    image: np.ndarray,
    masks: np.ndarray,
    config: dict | None = ...,
    *,
    image_id: str | int | None = ...,
    return_as: Literal["tuple"] = ...,
) -> tuple[np.ndarray, list[str], list[tuple]]: ...


@overload
def featurize(
    image: np.ndarray,
    masks: np.ndarray,
    config: dict | None = ...,
    *,
    image_id: str | int | None = ...,
    return_as: Literal["pandas"],
) -> pd.DataFrame: ...


@overload
def featurize(
    image: np.ndarray,
    masks: np.ndarray,
    config: dict | None = ...,
    *,
    image_id: str | int | None = ...,
    return_as: Literal["pyarrow"],
) -> pa.Table: ...


@overload
def featurize(
    image: np.ndarray,
    masks: np.ndarray,
    config: dict | None = ...,
    *,
    image_id: str | int | None = ...,
    return_as: Literal["anndata"],
) -> ad.AnnData: ...


def featurize(
    image: np.ndarray,
    masks: np.ndarray,
    config: dict | None = None,
    *,
    image_id: str | int | None = None,
    return_as: Literal["tuple", "pandas", "pyarrow", "anndata"] = "tuple",
):
    """Compute all configured features for the given image and masks.

    Parameters
    ----------
    image : numpy.ndarray
        Multichannel image with shape ``(C, H, W)`` or ``(C, Z, H, W)``.
    masks : numpy.ndarray
        Integer-labeled masks with shape ``(M, H, W)`` or
        ``(M, Z, H, W)``.  Must have the same ``ndim`` as *image*.
        Background is 0; labels must be contiguous integers ``1..N``
        (standard cp_measure convention, see
        ``skimage.segmentation.relabel_sequential``).
    config : dict, optional
        Configuration dictionary produced by :func:`make_featurizer_config`.
        If ``None``, all features are enabled with default parameters.
    image_id : str | int | None, optional
        Identifier for this image, stored in each row tuple.
    return_as : str, optional
        Output format.  One of ``"tuple"`` (default), ``"pandas"``,
        ``"pyarrow"``, or ``"anndata"``.  Non-tuple formats require the
        corresponding package to be installed (e.g.
        ``pip install cp_measure[anndata]``).

    Returns
    -------
    tuple or pd.DataFrame or pa.Table or anndata.AnnData
        When ``return_as="tuple"`` (default): ``(data, columns, rows)``
        where *data* is a 2-D float array, *columns* is a list of
        feature names, and *rows* is a list of
        ``(image_id, object_name, label)`` tuples.

        When ``return_as="pandas"``: a DataFrame with feature columns
        plus ``image_id``, ``object_type``, and ``label`` columns.

        When ``return_as="pyarrow"``: a PyArrow Table with per-column
        metadata in the schema.

        When ``return_as="anndata"``: an AnnData object with features
        in ``X``, object metadata in ``obs``, feature metadata in
        ``var``, and configuration in ``uns``.
    """
    _valid_return_as = {"tuple", "pandas", "pyarrow", "anndata"}
    if return_as not in _valid_return_as:
        raise ValueError(
            f"return_as must be one of {_valid_return_as!r}, got {return_as!r}"
        )
    if config is None:
        config = make_featurizer_config()
    channels, objects = _resolve_names(config, image.shape[0])
    _validate(image, masks, channels, objects)

    from cp_measure.bulk import get_core_measurements, get_correlation_measurements

    is_3d = image.ndim == 4
    core_funcs = get_core_measurements()
    corr_funcs = get_correlation_measurements()

    skipped_2d = _warn_and_filter_2d_only(config, is_3d)
    channel_feats = _collect_channel_features(config, core_funcs, skipped=skipped_2d)
    shape_feats = _collect_shape_features(config, core_funcs, skipped=skipped_2d)
    corr_feats = _collect_correlation_features(config, corr_funcs, len(channels))

    # Shape features are purely geometric and ignore pixel values.
    dummy_pixels = None
    collect_meta = return_as != "tuple"

    all_rows: list[tuple] = []
    all_blocks: list[np.ndarray] = []
    columns: list[str] | None = None
    col_meta: list[dict] | None = None

    for mask_idx, object_name in enumerate(objects):
        mask = masks[mask_idx]
        # Assumes contiguous labels 1..max (standard cp_measure contract).
        n_labels = int(mask.max())
        if n_labels == 0:
            continue

        results: dict[str, np.ndarray] = {}
        building_meta = collect_meta and columns is None
        meta_entries: list[dict] = []

        for func, params, group_name in shape_feats:
            raw = func(mask, dummy_pixels, **params)
            results.update(raw)
            if building_meta:
                for key in raw:
                    meta_entries.append(
                        _meta_entry(group_name, "shape", key)
                    )

        for ch_idx, ch_name in enumerate(channels):
            pixels = image[ch_idx]
            for func, params, group_name in channel_feats:
                raw = func(mask, pixels, **params)
                for key, values in raw.items():
                    results[f"{key}__{ch_name}"] = values
                    if building_meta:
                        meta_entries.append(
                            _meta_entry(group_name, "channel", key, channel=ch_name)
                        )

        n_ch = len(channels)
        for func, params, symmetric, group_name in corr_feats:
            iter_fn = itertools.combinations if symmetric else itertools.permutations
            for ch_i, ch_j in iter_fn(range(n_ch), 2):
                raw = func(
                    pixels_1=image[ch_i],
                    pixels_2=image[ch_j],
                    masks=mask,
                    **params,
                )
                for key, values in raw.items():
                    results[f"{key}__{channels[ch_i]}__{channels[ch_j]}"] = values
                    if building_meta:
                        meta_entries.append(
                            _meta_entry(
                                group_name,
                                "correlation",
                                key,
                                channel=channels[ch_i],
                                channel_2=channels[ch_j],
                            )
                        )

        # Build column list from the first non-empty mask.
        # Order-sensitive comparison is safe: all measurement functions
        # return plain dicts whose insertion order is deterministic in
        # Python 3.7+ and we iterate channels/pairs in the same order
        # for every mask.
        col_names = list(results.keys())
        if columns is None:
            columns = col_names
            if building_meta:
                col_meta = meta_entries
        elif col_names != columns:
            raise RuntimeError(
                f"feature keys for object {object_name!r} differ from "
                f"the first object — this is a bug in cp_measure"
            )

        block = np.column_stack([results[c] for c in columns])
        all_blocks.append(block)

        all_rows.extend(
            (image_id, object_name, label) for label in range(1, n_labels + 1)
        )

    if not all_blocks:
        raise ValueError("all masks have no labels (all zeros)")

    data = np.vstack(all_blocks)

    if return_as == "tuple":
        return data, columns, all_rows

    from cp_measure._converters import convert

    return convert(
        return_as,
        data=data,
        columns=columns,
        rows=all_rows,
        col_meta=col_meta,
        config=config,
        channels=channels,
        objects=objects,
        is_3d=is_3d,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _meta_entry(
    group: str, ftype: str, name: str, *, channel: str | None = None, channel_2: str | None = None
) -> dict:
    """Build a per-column metadata dict for ``var`` / schema metadata."""
    return {
        "feature_group": group,
        "feature_type": ftype,
        "feature_name": name,
        "channel": channel,
        "channel_2": channel_2,
    }


def _resolve_channels(n_channels: int) -> list[str]:
    """Generate default channel names, zero-padded when n >= 10."""
    width = len(str(n_channels - 1)) if n_channels >= 10 else 1
    return [f"ch{i:0{width}d}" for i in range(n_channels)]


def _resolve_names(config: dict, n_image_channels: int) -> tuple[list[str], list[str]]:
    """Resolve channel and object names from config, warning if auto-named."""
    channels = config["channels"]
    if channels is None:
        channels = _resolve_channels(n_image_channels)
        warnings.warn(
            "No channel names provided — auto-assigning "
            f"{channels}. Consider passing explicit channel names "
            "for reproducibility.",
            UserWarning,
            stacklevel=3,
        )
    objects = config["objects"]
    return channels, objects


def _validate(
    image: np.ndarray,
    masks: np.ndarray,
    channels: list[str],
    objects: list[str],
) -> None:
    """Validate image and mask inputs."""
    if image.ndim not in (3, 4):
        raise ValueError(
            f"image must be 3D (C, H, W) or 4D (C, Z, H, W), got shape {image.shape}"
        )
    if masks.ndim not in (3, 4):
        raise ValueError(
            f"masks must be 3D (M, H, W) or 4D (M, Z, H, W), got shape {masks.shape}"
        )
    if image.ndim != masks.ndim:
        raise ValueError(
            f"image and masks must have the same number of dimensions, "
            f"got image.ndim={image.ndim} and masks.ndim={masks.ndim}"
        )
    if channels and image.shape[0] != len(channels):
        raise ValueError(
            f"image has {image.shape[0]} channels but "
            f"{len(channels)} channel names were provided"
        )
    if masks.shape[0] != len(objects):
        raise ValueError(
            f"masks has {masks.shape[0]} object masks but "
            f"{len(objects)} object names were provided"
        )
    if image.shape[1:] != masks.shape[1:]:
        raise ValueError(
            f"spatial dims mismatch: image {image.shape[1:]}, masks {masks.shape[1:]}"
        )
    if not np.issubdtype(masks.dtype, np.integer):
        raise TypeError(f"masks must be integer dtype, got {masks.dtype}")


def _collect_channel_features(
    config: dict, core_funcs: dict, *, skipped: set[str]
) -> list[tuple]:
    """Collect enabled per-channel feature functions and their params.

    Each element is ``(func, params, group_name)``.
    """
    feats: list[tuple] = []
    for name in (
        "intensity",
        "texture",
        "granularity",
        "radial_distribution",
        "radial_zernikes",
    ):
        if config[name] and name not in skipped:
            feats.append((core_funcs[name], config[f"{name}_params"], name))
    return feats


def _collect_shape_features(
    config: dict, core_funcs: dict, *, skipped: set[str]
) -> list[tuple]:
    """Collect enabled shape feature functions and their params.

    Each element is ``(func, params, group_name)``.
    """
    feats: list[tuple] = []
    for name in ("sizeshape", "zernike"):
        if config[name] and name not in skipped:
            feats.append((core_funcs[name], config[f"{name}_params"], name))
    if config["feret"] and "feret" not in skipped:
        feats.append((core_funcs["feret"], {}, "feret"))
    return feats


def _warn_and_filter_2d_only(config: dict, is_3d: bool) -> set[str]:
    """Return the set of 2D-only feature names to skip, warning if any."""
    if not is_3d:
        return set()
    skipped = {name for name in _2D_ONLY if config.get(name, False)}
    if skipped:
        warnings.warn(
            f"Skipping 2D-only features for volumetric data: {sorted(skipped)}",
            UserWarning,
            stacklevel=2,
        )
    return skipped


def _collect_correlation_features(
    config: dict,
    corr_funcs: dict,
    n_channels: int,
) -> list[tuple]:
    """Collect enabled correlation feature functions.

    The third element of each tuple indicates whether the metric is
    symmetric (combinations) or asymmetric (permutations).  The fourth
    element is the feature group name.
    """
    if n_channels < 2:
        has_corr = any(
            config[k]
            for k in (
                "correlation_pearson",
                "correlation_costes",
                "correlation_manders_fold",
                "correlation_rwc",
            )
        )
        if has_corr:
            warnings.warn(
                "correlation features require at least 2 channels; "
                "skipping correlation since only 1 channel was provided",
                UserWarning,
                stacklevel=3,
            )
        return []

    feats: list[tuple] = []
    # (config key, corr_funcs key, params key, symmetric)
    specs = [
        ("correlation_pearson", "pearson", None, False),
        ("correlation_costes", "costes", "correlation_costes_params", False),
        (
            "correlation_manders_fold",
            "manders_fold",
            "correlation_manders_fold_params",
            True,
        ),
        ("correlation_rwc", "rwc", "correlation_rwc_params", True),
    ]
    for cfg_key, func_key, params_key, symmetric in specs:
        if config[cfg_key]:
            params = config[params_key] if params_key else {}
            feats.append((corr_funcs[func_key], params, symmetric, cfg_key))
    return feats
