"""High-level featurization wrapper for cp_measure.

Provides two stateless functions:

- :func:`make_featurizer_config` builds a plain configuration dictionary.
- :func:`featurize` takes that configuration together with image and mask
  arrays and returns a numpy feature matrix with column and row metadata.

``featurize`` takes images shaped ``(B, C, *spatial)`` and masks shaped
``(B, M, *spatial)``: B is the batch size, C the channel count, and M the number
of mask types. A single image still needs a batch axis, with ``B == 1``.
Dimensionality is declared with ``is_3d``, never inferred from shape. Images and
masks must have matching spatial dimensions; no axes are broadcast automatically.

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
from collections.abc import Callable
from typing import cast

import numpy as np

from cp_measure._sanitize import sanitize_masks
from cp_measure.primitives.parallel import map_images

# Feature groups that only support 2D spatial data.
_Measurement = Callable[..., list[dict[str, np.ndarray]]]  # batch in, one dict per item
_Feature = tuple[_Measurement, dict]  # measurement function, keyword arguments
_CorrelationFeature = tuple[_Measurement, dict, bool]  # plus symmetric-pair flag


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
        Configuration dictionary accepted by :func:`featurize`. Granularity,
        Manders and RWC parameter dictionaries explicitly record the default
        ``precision="input"``; set ``precision="float64"`` in each feature's
        params to opt in. Save the config with results for reproducibility.
        Caller-owned parameter dictionaries are not modified.

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
        "granularity_params": {"precision": "input", **(granularity_params or {})},
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
        "correlation_manders_fold_params": {
            "precision": "input",
            **(correlation_manders_fold_params or {}),
        },
        "correlation_rwc": correlation_rwc,
        "correlation_rwc_params": {
            "precision": "input",
            **(correlation_rwc_params or {}),
        },
    }


def _featurize_batch(
    image: np.ndarray,
    masks: np.ndarray,
    batch_idx: list[int],
    image_ids: list,
    channels: list[str],
    objects: list[str],
    shape_feats: list[_Feature],
    channel_feats: list[_Feature],
    corr_feats: list[_CorrelationFeature],
) -> tuple[dict[int, tuple[list[np.ndarray], list[tuple]]], list[str]]:
    """Measure the batch items ``batch_idx`` with pre-resolved names and features.

    Internal worker: all setup (config/name resolution, feature collection,
    user-facing warnings) is done once by the public batcher :func:`featurize`.
    For each mask type, every channel feature runs ONE call over all images and
    channels, and every correlation feature ONE call over all images and channel
    pairs, so a backend can parallelize across the whole batch and reuse per-mask
    work. Returns ``({b: (blocks, rows)}, columns)``: one block per non-empty mask
    type of each item, with the same columns and order as measuring each image on
    its own.

    ``shape_feats`` and ``channel_feats`` contain ``(function, kwargs)`` pairs:
    shape measurements run once per mask type, channel measurements once per mask
    type and channel. ``corr_feats`` contains ``(function, kwargs, symmetric)``
    triples. Symmetric correlations measure each channel pair once; asymmetric
    correlations measure both channel orders.
    """
    n_ch = len(channels)
    out: dict[int, tuple[list[np.ndarray], list[tuple]]] = {
        b: ([], []) for b in batch_idx
    }
    columns: list[str] | None = None

    for mask_idx, object_name in enumerate(objects):
        # Relabel arbitrary IDs to 1..N once up front; keep originals for rows.
        # Relabelled on the thread pool: one image after another on the calling
        # thread left the workers idle (sort and relabel release the GIL).
        sanitized = map_images(
            lambda b: sanitize_masks(masks[b, mask_idx]),
            batch_idx,
            pixels=len(batch_idx) * masks[0, mask_idx].size,
        )
        items = [
            (b, clean, ids) for b, (clean, ids) in zip(batch_idx, sanitized) if ids.size
        ]
        if not items:
            continue
        results: list[dict[str, np.ndarray]] = [{} for _ in items]

        # Shape features are purely geometric and ignore pixel values. One call
        # per feature over the whole batch.
        for func, params in shape_feats:
            outputs = func([clean for _, clean, _ in items], None, **params)
            for result, output in zip(results, outputs):
                result.update(output)

        # One call per channel feature, returned channel-major (image i, channel c).
        # Reshape each flat result to one row of n_ch per image, so the scatter
        # indexes [i][c] explicitly instead of computing i * n_ch + c; strict=True
        # fails if the backend returned the wrong count. Columns stay channel-major.
        per_feature = [
            list(
                itertools.batched(
                    func(
                        [clean for _, clean, _ in items for _ in range(n_ch)],
                        [image[b, c] for b, _, _ in items for c in range(n_ch)],
                        **params,
                    ),
                    n_ch,
                    strict=True,
                )
            )
            for func, params in (channel_feats if n_ch else [])
        ]
        for i, result in enumerate(results):
            for c, ch_name in enumerate(channels):
                for rows in per_feature:
                    for key, values in rows[i][c].items():
                        result[f"{key}__{ch_name}"] = values

        # One call per correlation feature, returned pair-major (image i, pair p);
        # reshaped to one row of len(pairs) per image so the scatter indexes [i][p].
        for func, params, symmetric in corr_feats:
            iter_fn = itertools.combinations if symmetric else itertools.permutations
            pairs = list(iter_fn(range(n_ch), 2))
            if not pairs:
                continue
            rows = list(
                itertools.batched(
                    func(
                        pixels_1=[image[b, c] for b, _, _ in items for c, _ in pairs],
                        pixels_2=[image[b, c] for b, _, _ in items for _, c in pairs],
                        masks=[clean for _, clean, _ in items for _ in pairs],
                        **params,
                    ),
                    len(pairs),
                    strict=True,
                )
            )
            for i, result in enumerate(results):
                for p, (ch_i, ch_j) in enumerate(pairs):
                    for key, values in rows[i][p].items():
                        result[f"{key}__{channels[ch_i]}__{channels[ch_j]}"] = values

        # Every item uses the same feature lists, and all measurement functions
        # return plain dicts in a deterministic order, so the columns must match.
        for result, (b, _, ids) in zip(results, items):
            if columns is None:
                columns = list(result)
            elif list(result) != columns:
                raise RuntimeError(
                    f"feature keys for object {object_name!r} differ from "
                    f"the first object — this is a bug in cp_measure"
                )
            blocks, rows = out[b]
            # One feature per row of a preallocated array, then transposed:
            # column_stack over ~1000 columns was a large share of the time the
            # calling thread spends with the GIL on a many-core batch.
            block = np.empty((len(columns), len(ids)))
            for j, key in enumerate(columns):
                if np.ndim(result[key]) != 1 or len(result[key]) != len(ids):
                    raise RuntimeError(
                        f"feature {key!r} has the wrong length for object "
                        f"{object_name!r} — this is a bug in cp_measure"
                    )
                block[j] = result[key]
            blocks.append(block.T)
            rows.extend(
                zip(itertools.repeat(image_ids[b]), itertools.repeat(object_name), ids)
            )
    # featurize only calls this with at least one labelled image.
    return out, cast(list[str], columns)


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
    masks are ``(B, M, *spatial)`` with matching rank and spatial dimensions.
    A single image is ``B == 1``.

    Image dimensionality is *declared* via ``is_3d``, never inferred from shape, so
    a single-channel volume ``(1, 1, Z, Y, X)`` is never mistaken for a multichannel
    2D image. To use a 2D mask through a volume, the caller must explicitly repeat
    it along Z before calling this function. Inputs are not modified. For Z=1,
    2D-only families may measure the sole slice; for Z>1 they are skipped. This
    exception does not change the spatial rank passed to 3D-capable features.
    Size/shape also uses 2D features for Z=1; other families keep their own
    dimensional conventions.

    Images whose mask stack has no labelled objects are skipped before any
    measurement runs; the number skipped is reported via a warning.

    Parameters
    ----------
    image : numpy.ndarray
        ``(B, C, Y, X)`` or ``(B, C, Z, Y, X)`` (see ``is_3d``).
    masks : numpy.ndarray
        ``(B, M, Y, X)`` or ``(B, M, Z, Y, X)``; integer labels, background 0.
        With ``is_3d=True``, all five axes are required. Batch size and spatial
        dimensions must match the image exactly; no axes are broadcast.
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
    _validate_batch_inputs(image, masks, is_3d)
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
        _2D_ONLY,
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
        if image.shape[2] == 1:
            # Only the 2D-only families opt into sole-slice calculations. Keep
            # full spatial rank for genuinely 3D features and correlation.
            all_core_funcs = get_core_measurements(legacy=legacy, sanitize=False)
            core_funcs.update({name: all_core_funcs[name] for name in _2D_ONLY})
        else:
            _warn_2d_only_in_3d(config)
    shape_feats = _collect_shape_features(config, core_funcs)
    channel_feats = _collect_channel_features(config, core_funcs)
    corr_feats = _collect_correlation_features(config, corr_funcs, len(channels))

    # Empty mask stacks are skipped before the hot path.
    labelled = [bool(masks[b].any()) for b in range(batch_size)]
    skipped = [b for b in range(batch_size) if not labelled[b]]
    batch_idx = [b for b in range(batch_size) if labelled[b]]
    ids = image_ids if image_ids is not None else list(range(batch_size))
    measured, columns = _featurize_batch(
        image,
        masks,
        batch_idx,
        ids,
        channels,
        objects,
        shape_feats,
        channel_feats,
        corr_feats,
    )
    # Rows stay image-major, then mask type, as when measuring image by image.
    all_blocks = [block for b in batch_idx for block in measured[b][0]]
    all_rows = [row for b in batch_idx for row in measured[b][1]]

    if not all_blocks:
        raise ValueError("no images had labelled objects (all batch items were empty)")
    if skipped:
        warnings.warn(
            f"{len(skipped)} of {batch_size} image(s) had no labelled objects and were "
            f"skipped (batch indices {skipped}).",
            UserWarning,
            stacklevel=2,
        )

    # One C-ordered matrix; the blocks are transposed views, copied into it once.
    data = np.empty((sum(len(block) for block in all_blocks), len(columns)))
    start = 0
    for block in all_blocks:
        data[start : start + len(block)] = block
        start += len(block)
    return data, columns, all_rows


def _validate_batch_inputs(image: np.ndarray, masks: np.ndarray, is_3d: bool) -> None:
    """Check batch/channel axes, matching spatial dimensions, and integer labels."""
    expected_ndim = 5 if is_3d else 4
    img_shape = "(B, C, Z, Y, X)" if is_3d else "(B, C, Y, X)"
    mask_shape = "(B, M, Z, Y, X)" if is_3d else "(B, M, Y, X)"
    if image.ndim != expected_ndim:
        hint = ""
        if image.ndim == 4:
            hint = (
                " For a 2D batch (B, C, Y, X), pass is_3d=False."
                " For 3D data, keep is_3d=True and supply all five axes."
            )
        elif image.ndim == 5:
            hint = " For a 3D batch (B, C, Z, Y, X), pass is_3d=True."
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
    :func:`_validate_batch_inputs`; this only checks that axis 1 matches the
    number of channel names (image) or mask-type names (masks).
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


def _collect_channel_features(config: dict, core_funcs: dict) -> list[_Feature]:
    """Collect enabled per-channel feature functions and their params.

    Features missing from ``core_funcs`` (e.g. 2D-only features when the
    registry is the 3D one) are silently skipped.
    """
    feats: list[_Feature] = []
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


def _collect_shape_features(config: dict, core_funcs: dict) -> list[_Feature]:
    """Collect enabled shape feature functions and their params.

    Features missing from ``core_funcs`` (e.g. 2D-only features when the
    registry is the 3D one) are silently skipped.
    """
    feats: list[_Feature] = []
    for name in ("sizeshape", "zernike"):
        if config[name] and name in core_funcs:
            feats.append((core_funcs[name], config[f"{name}_params"]))
    if config["feret"] and "feret" in core_funcs:
        feats.append((core_funcs["feret"], {}))
    return feats


def _warn_2d_only_in_3d(config: dict) -> None:
    """Warn if the config enables 2D-only features for a multi-slice volume."""
    from cp_measure.bulk import _2D_ONLY

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
) -> list[_CorrelationFeature]:
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

    feats: list[_CorrelationFeature] = []
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
