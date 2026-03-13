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

import numpy as np

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
        if len(channels) == 0:
            raise ValueError("channels must be a non-empty list when provided")
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


def featurize(
    image: np.ndarray,
    masks: np.ndarray,
    config: dict | None = None,
    *,
    image_id: str | int | None = None,
) -> tuple[np.ndarray, list[str], list[tuple]]:
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

    Returns
    -------
    data : numpy.ndarray
        2-D float array of shape ``(n_rows, n_features)``.
    columns : list[str]
        Feature column names.  Shape features are bare names (e.g.
        ``"Area"``), per-channel features are ``"{feature}__{channel}"``,
        and correlation features are ``"{feature}__{ch1}__{ch2}"``.
    rows : list[tuple]
        One ``(image_id, object_name, label)`` tuple per row.
    """
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

    all_rows: list[tuple] = []
    all_blocks: list[np.ndarray] = []
    columns: list[str] | None = None

    for mask_idx, object_name in enumerate(objects):
        mask = masks[mask_idx]
        # Assumes contiguous labels 1..max (standard cp_measure contract).
        n_labels = int(mask.max())
        if n_labels == 0:
            continue

        results: dict[str, np.ndarray] = {}

        for func, params in shape_feats:
            results.update(func(mask, dummy_pixels, **params))

        for ch_idx, ch_name in enumerate(channels):
            pixels = image[ch_idx]
            for func, params in channel_feats:
                for key, values in func(mask, pixels, **params).items():
                    results[f"{key}__{ch_name}"] = values

        n_ch = len(channels)
        for func, params, symmetric in corr_feats:
            iter_fn = itertools.combinations if symmetric else itertools.permutations
            for ch_i, ch_j in iter_fn(range(n_ch), 2):
                for key, values in func(
                    pixels_1=image[ch_i],
                    pixels_2=image[ch_j],
                    masks=mask,
                    **params,
                ).items():
                    results[f"{key}__{channels[ch_i]}__{channels[ch_j]}"] = values

        # Build column list from the first non-empty mask.
        # Order-sensitive comparison is safe: all measurement functions
        # return plain dicts whose insertion order is deterministic in
        # Python 3.7+ and we iterate channels/pairs in the same order
        # for every mask.
        col_names = list(results.keys())
        if columns is None:
            columns = col_names
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
    return data, columns, all_rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
    if image.shape[0] != len(channels):
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
    """Collect enabled per-channel feature functions and their params."""
    feats: list[tuple] = []
    for name in (
        "intensity",
        "texture",
        "granularity",
        "radial_distribution",
        "radial_zernikes",
    ):
        if config[name] and name not in skipped:
            feats.append((core_funcs[name], config[f"{name}_params"]))
    return feats


def _collect_shape_features(
    config: dict, core_funcs: dict, *, skipped: set[str]
) -> list[tuple]:
    """Collect enabled shape feature functions and their params."""
    feats: list[tuple] = []
    for name in ("sizeshape", "zernike"):
        if config[name] and name not in skipped:
            feats.append((core_funcs[name], config[f"{name}_params"]))
    if config["feret"] and "feret" not in skipped:
        feats.append((core_funcs["feret"], {}))
    return feats


def _warn_and_filter_2d_only(config: dict, is_3d: bool) -> set[str]:
    """Return the set of 2D-only feature names to skip, warning if any."""
    if not is_3d:
        return set()
    return {name for name in _2D_ONLY if config.get(name, False)}


def _collect_correlation_features(
    config: dict,
    corr_funcs: dict,
    n_channels: int,
) -> list[tuple]:
    """Collect enabled correlation feature functions.

    The third element of each tuple indicates whether the metric is
    symmetric (combinations) or asymmetric (permutations).
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
            feats.append((corr_funcs[func_key], params, symmetric))
    return feats
