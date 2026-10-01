"""High-level featurization wrapper for cp_measure.

Provides two stateless functions:

- :func:`make_featurizer_config` builds a plain configuration dictionary.
- :func:`featurize` takes that configuration together with image and mask
  arrays and returns a numpy feature matrix with column and row metadata.

``featurize`` is the single public entry point and takes a batch in the canonical
shape ``(B, C, *spatial)`` (image) and ``(B, M, *spatial)`` (masks); a single image
is ``B == 1``. Dimensionality is declared with ``is_3d``, never inferred from shape.

Example
-------
>>> from cp_measure.featurizer import make_featurizer_config, featurize
>>> config = make_featurizer_config(["DNA", "ER"], objects=["nuclei", "cells"])
>>> # image: (B, 2, Y, X)   masks: (B, 2, Y, X)
>>> data, columns, rows = featurize(image, masks, config)  # is_3d=False (2-D)
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np

from cp_measure._sanitize import sanitize_masks

# Feature groups that only support 2D spatial data.
_2D_ONLY = {"radial_distribution", "radial_zernikes", "zernike", "feret"}


def make_featurizer_config(
    channels: list[str] | None = None,
    *,
    objects: list[str] | None = None,
    legacy: bool = False,
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
        "legacy": legacy,
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


def _featurize_one(
    image: np.ndarray,
    masks: np.ndarray,
    channels: list[str],
    objects: list[str],
    shape_feats: list[tuple],
    channel_feats: list[tuple],
    corr_feats: list[tuple],
    *,
    image_id: str | int | None = None,
) -> tuple[np.ndarray, list[str], list[tuple]]:
    """Measure one image ``(C, *spatial)`` with pre-resolved names and feature lists.

    Internal per-item worker: all setup (config/name resolution, feature collection,
    user-facing warnings) is done once by the public batcher :func:`featurize`, so this
    runs only the per-object measurement loop. Returns ``(data, columns, rows)`` as
    described on :func:`featurize`.
    """
    # Shape features are purely geometric and ignore pixel values.
    dummy_pixels = None

    all_rows: list[tuple] = []
    all_blocks: list[np.ndarray] = []
    columns: list[str] | None = None

    for mask_idx, object_name in enumerate(objects):
        # Relabel arbitrary IDs to 1..N once up front; keep originals for rows.
        clean, ids = sanitize_masks(masks[mask_idx])
        if ids.size == 0:
            continue

        results: dict[str, np.ndarray] = {}

        for func, params in shape_feats:
            results.update(func(clean, dummy_pixels, **params))

        for ch_idx, ch_name in enumerate(channels):
            pixels = image[ch_idx]
            for func, params in channel_feats:
                for key, values in func(clean, pixels, **params).items():
                    results[f"{key}__{ch_name}"] = values

        n_ch = len(channels)
        for func, params, symmetric in corr_feats:
            iter_fn = itertools.combinations if symmetric else itertools.permutations
            for ch_i, ch_j in iter_fn(range(n_ch), 2):
                for key, values in func(
                    pixels_1=image[ch_i],
                    pixels_2=image[ch_j],
                    masks=clean,
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

        all_rows.extend((image_id, object_name, label) for label in ids)

    return np.vstack(all_blocks), columns, all_rows


def featurize(
    image: np.ndarray,
    masks: np.ndarray,
    config: dict | None = None,
    *,
    is_3d: bool = False,
    image_ids: list[str | int] | None = None,
) -> tuple[np.ndarray, list[str], list[tuple]]:
    """Compute all configured features for a batch of images.

    The single public entry point. Input is strictly ``(B, C, *spatial)``:
    ``(B, C, Y, X)`` when ``is_3d=False`` and ``(B, C, Z, Y, X)`` when ``is_3d=True``;
    masks are ``(B, M, *spatial)`` with matching rank. A single image is ``B == 1``.

    Dimensionality is *declared* via ``is_3d``, never inferred from shape, so a
    single-channel volume ``(1, 1, Z, Y, X)`` is never mistaken for a multichannel
    2-D image. Rank is validated against ``is_3d`` and any deviation is a hard error.

    Images whose mask stack has no labelled objects are skipped before any
    measurement runs; the number skipped is reported via a warning.

    Parameters
    ----------
    image : numpy.ndarray
        ``(B, C, Y, X)`` or ``(B, C, Z, Y, X)`` (see ``is_3d``).
    masks : numpy.ndarray
        ``(B, M, Y, X)`` or ``(B, M, Z, Y, X)``; integer labels, background 0.
    config : dict, optional
        As :func:`make_featurizer_config`; defaults to all features enabled.
    is_3d : bool, default False
        Declare 3-D input (adds the ``Z`` axis). Rank is validated against this.
    image_ids : list, optional
        One identifier per batch item, stored in each row. Defaults to the batch
        index; pass caller-unique ids to keep provenance across multiple calls.

    Returns
    -------
    data : numpy.ndarray
        2-D float array ``(n_rows, n_features)`` stacked over all non-empty images.
    columns : list[str]
        Feature column names (identical across batch items).
    rows : list[tuple]
        One ``(image_id, object_name, label)`` per row.
    """
    _validate_canonical(image, masks, is_3d)
    batch_size = image.shape[0]
    if image_ids is not None and len(image_ids) != batch_size:
        raise ValueError(
            f"image_ids has {len(image_ids)} entries but batch has {batch_size} images"
        )

    # Resolve config, names and feature lists ONCE — they are constant across the batch,
    # so user-facing warnings fire a single time and setup is not repeated per item.
    if config is None:
        config = make_featurizer_config()
    channels, objects = _resolve_names(config, image.shape[1])
    _validate_names(image, masks, channels, objects)

    from cp_measure.bulk import (
        get_core_measurements,
        get_core_measurements_3d,
        get_correlation_measurements,
    )

    legacy = config.get("legacy", False)
    # Sanitize each mask once per item below, so fetch raw (unsanitized) funcs.
    core_funcs = (
        get_core_measurements_3d(legacy=legacy, sanitize=False)
        if is_3d
        else get_core_measurements(legacy=legacy, sanitize=False)
    )
    corr_funcs = get_correlation_measurements(sanitize=False)
    if is_3d:
        _warn_2d_only_in_3d(config)
    shape_feats = _collect_shape_features(config, core_funcs)
    channel_feats = _collect_channel_features(config, core_funcs)
    corr_feats = _collect_correlation_features(config, corr_funcs, len(channels))

    all_blocks: list[np.ndarray] = []
    all_rows: list[tuple] = []
    columns: list[str] | None = None
    skipped: list[int] = []

    for b in range(batch_size):
        if not masks[b].any():  # empty mask stack: skip before the hot path
            skipped.append(b)
            continue
        image_id = image_ids[b] if image_ids is not None else b
        data, cols, rows = _featurize_one(
            image[b],
            masks[b],
            channels,
            objects,
            shape_feats,
            channel_feats,
            corr_feats,
            image_id=image_id,
        )
        # Every item is measured with the same feature lists, so columns are
        # identical by construction (per-object consistency is guarded in
        # _featurize_one); just capture them from the first non-empty item.
        if columns is None:
            columns = cols
        all_blocks.append(data)
        all_rows.extend(rows)

    if not all_blocks:
        raise ValueError("no images had labelled objects (all batch items were empty)")
    if skipped:
        warnings.warn(
            f"{len(skipped)} of {batch_size} image(s) had no labelled objects and were "
            f"skipped (batch indices {skipped}).",
            UserWarning,
            stacklevel=2,
        )

    return np.vstack(all_blocks), columns, all_rows


def _validate_canonical(image: np.ndarray, masks: np.ndarray, is_3d: bool) -> None:
    """Enforce the strict ``(B, C, *spatial)`` contract; raise on any deviation."""
    expected_ndim = 5 if is_3d else 4
    img_shape = "(B, C, Z, Y, X)" if is_3d else "(B, C, Y, X)"
    mask_shape = "(B, M, Z, Y, X)" if is_3d else "(B, M, Y, X)"
    if image.ndim != expected_ndim:
        hint = ""
        if image.ndim == 4:
            hint = " Pass is_3d=False for 2-D data."
        elif image.ndim == 5:
            hint = " Pass is_3d=True for a 3-D volume."
        raise ValueError(
            f"image must be {img_shape} for is_3d={is_3d} (ndim {expected_ndim}), "
            f"got ndim {image.ndim} with shape {image.shape}.{hint}"
        )
    if masks.ndim != expected_ndim:
        raise ValueError(
            f"masks must be {mask_shape} for is_3d={is_3d} (ndim {expected_ndim}), "
            f"got ndim {masks.ndim} with shape {masks.shape}"
        )
    if image.shape[0] != masks.shape[0]:
        raise ValueError(
            f"batch size mismatch: image has {image.shape[0]} items, "
            f"masks has {masks.shape[0]}"
        )
    if image.shape[2:] != masks.shape[2:]:
        raise ValueError(
            f"spatial dims mismatch: image {image.shape[2:]}, masks {masks.shape[2:]}"
        )
    if not np.issubdtype(masks.dtype, np.integer):
        raise TypeError(f"masks must be integer dtype, got {masks.dtype}")


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


def _validate_names(
    image: np.ndarray,
    masks: np.ndarray,
    channels: list[str],
    objects: list[str],
) -> None:
    """Check channel/object counts against the provided names.

    Rank, spatial dims, batch size and dtype are already enforced by
    :func:`_validate_canonical`; this only checks the name counts (``C`` and ``M``,
    the axis-1 sizes of the canonical ``(B, C|M, *spatial)`` arrays).
    """
    if channels and image.shape[1] != len(channels):
        raise ValueError(
            f"image has {image.shape[1]} channels but "
            f"{len(channels)} channel names were provided"
        )
    if masks.shape[1] != len(objects):
        raise ValueError(
            f"masks has {masks.shape[1]} object masks but "
            f"{len(objects)} object names were provided"
        )


def _collect_channel_features(config: dict, core_funcs: dict) -> list[tuple]:
    """Collect enabled per-channel feature functions and their params.

    Features missing from ``core_funcs`` (e.g. 2D-only features when the
    registry is the 3D one) are silently skipped.
    """
    feats: list[tuple] = []
    for name in (
        "intensity",
        "texture",
        "granularity",
        "radial_distribution",
        "radial_zernikes",
    ):
        if config[name] and name in core_funcs:
            feats.append((core_funcs[name], config[f"{name}_params"]))
    return feats


def _collect_shape_features(config: dict, core_funcs: dict) -> list[tuple]:
    """Collect enabled shape feature functions and their params.

    Features missing from ``core_funcs`` (e.g. 2D-only features when the
    registry is the 3D one) are silently skipped.
    """
    feats: list[tuple] = []
    for name in ("sizeshape", "zernike"):
        if config[name] and name in core_funcs:
            feats.append((core_funcs[name], config[f"{name}_params"]))
    if config["feret"] and "feret" in core_funcs:
        feats.append((core_funcs["feret"], {}))
    return feats


def _warn_2d_only_in_3d(config: dict) -> None:
    """Warn if the config enables any 2D-only feature for 3D input."""
    requested = sorted(name for name in _2D_ONLY if config.get(name, False))
    if requested:
        warnings.warn(
            f"3D input — skipping 2D-only feature(s): {requested}",
            UserWarning,
            stacklevel=3,
        )


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
