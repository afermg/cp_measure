"""High-level featurization orchestrator for cp_measure."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

# Mapping from make_featurizer kwarg names to bulk.py registry keys
_CHANNEL_FEATURES = {
    "intensity": "intensity",
    "texture": "texture",
    "granularity": "granularity",
    "radial_distribution": "radial_distribution",
    "radial_zernikes": "radial_zernikes",
}

_SHAPE_FEATURES = {
    "sizeshape": "sizeshape",
    "zernike": "zernike",
    "ferret": "ferret",
}

# Correlation features: (registry_key, symmetric).
# Symmetric metrics produce identical results for (A, B) and (B, A) —
# either directly or via _1/_2 suffix swap — so combinations suffice.
# Asymmetric metrics must be run for both orderings (permutations).
_CORRELATION_FEATURES = {
    "correlation_pearson": ("pearson", False),
    "correlation_costes": ("costes", False),
    "correlation_manders_fold": ("manders_fold", True),
    "correlation_rwc": ("rwc", True),
}


class Featurizer:
    """Configured featurization pipeline.

    Use :func:`make_featurizer` to create instances.

    Parameters
    ----------
    channels : list[str]
        Channel names corresponding to the first axis of the image array.
    masks : list[str]
        Mask names corresponding to the first axis of the masks array.
    channel_features : list[tuple[str, callable, dict]]
        Per-channel features as ``(name, func, params)`` tuples.
    shape_features : list[tuple[str, callable, dict]]
        Mask-only shape features as ``(name, func, params)`` tuples.
    correlation_features : list[tuple[str, callable, dict, bool]]
        Pairwise correlation features as ``(name, func, params, symmetric)``
        tuples. Symmetric metrics use combinations; asymmetric use
        permutations.
    """

    def __init__(
        self,
        channels: list[str],
        masks: list[str],
        channel_features: list[tuple[str, callable, dict]],
        shape_features: list[tuple[str, callable, dict]],
        correlation_features: list[tuple[str, callable, dict, bool]],
    ):
        self._channels = list(channels)
        self._masks = list(masks)
        self._channel_features = channel_features
        self._shape_features = shape_features
        self._correlation_features = correlation_features

    def featurize(self, image: np.ndarray, masks: np.ndarray) -> pd.DataFrame:
        """Compute all configured features for the given image and masks.

        Parameters
        ----------
        image : numpy.ndarray
            Multichannel image with shape ``(C, H, W)`` where ``C`` matches
            the number of channel names provided at construction time.
        masks : numpy.ndarray
            Integer-labeled masks with shape ``(M, H, W)`` where ``M`` matches
            the number of mask names provided at construction time. Background
            is 0, each object has a unique positive integer label per plane.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per labeled object (union of all labels
            across masks). Index is the label ID, columns are named
            ``{mask}_{feature}_{channel}`` for per-channel features,
            ``{mask}_{feature}`` for shape features, and
            ``{mask}_{feature}_{ch1}_{ch2}`` for correlation features.
            Labels missing from a given mask have NaN for that mask's columns.
        """
        self._validate(image, masks)

        per_mask_dfs: list[pd.DataFrame] = []

        for mask_idx, mask_name in enumerate(self._masks):
            mask_2d = masks[mask_idx]
            n_labels = mask_2d.max()

            if n_labels == 0:
                # This mask plane has no labels; skip it
                continue

            results: dict[str, np.ndarray] = {}

            # Shape features — run once per mask, no channel suffix
            for _name, func, params in self._shape_features:
                result = func(mask_2d, image[0], **params)
                for key, values in result.items():
                    results[f"{mask_name}_{key}"] = values

            # Per-channel intensity features
            for ch_idx, ch_name in enumerate(self._channels):
                pixels = image[ch_idx]
                for _name, func, params in self._channel_features:
                    result = func(mask_2d, pixels, **params)
                    for key, values in result.items():
                        results[f"{mask_name}_{key}_{ch_name}"] = values

            # Correlation features — symmetric use combinations, asymmetric
            # use permutations to capture both channel orderings
            n_ch = len(self._channels)
            for _name, func, params, symmetric in self._correlation_features:
                pairs = (
                    itertools.combinations(range(n_ch), 2)
                    if symmetric
                    else itertools.permutations(range(n_ch), 2)
                )
                for ch_i, ch_j in pairs:
                    name_i = self._channels[ch_i]
                    name_j = self._channels[ch_j]
                    result = func(
                        pixels_1=image[ch_i],
                        pixels_2=image[ch_j],
                        masks=mask_2d,
                        **params,
                    )
                    for key, values in result.items():
                        results[f"{mask_name}_{key}_{name_i}_{name_j}"] = values

            mask_df = pd.DataFrame(results, index=np.arange(1, n_labels + 1))
            per_mask_dfs.append(mask_df)

        # Outer-join all per-mask DataFrames on label index
        if not per_mask_dfs:
            raise ValueError("all mask planes have no labels (all zeros)")
        df = pd.concat(per_mask_dfs, axis=1, join="outer")

        df.index.name = "label"
        return df

    def _validate(self, image: np.ndarray, masks: np.ndarray) -> None:
        """Validate inputs before featurization."""
        if image.ndim != 3:
            raise ValueError(f"image must be 3D (C, H, W), got shape {image.shape}")
        if masks.ndim != 3:
            raise ValueError(f"masks must be 3D (M, H, W), got shape {masks.shape}")
        if image.shape[0] != len(self._channels):
            raise ValueError(
                f"image has {image.shape[0]} channels but "
                f"{len(self._channels)} channel names were provided"
            )
        if masks.shape[0] != len(self._masks):
            raise ValueError(
                f"masks has {masks.shape[0]} planes but "
                f"{len(self._masks)} mask names were provided"
            )
        if image.shape[1:] != masks.shape[1:]:
            raise ValueError(
                f"spatial dims mismatch: image {image.shape[1:]}, "
                f"masks {masks.shape[1:]}"
            )
        if not np.issubdtype(masks.dtype, np.integer):
            raise TypeError(f"masks must be integer dtype, got {masks.dtype}")
        if masks.max() == 0:
            raise ValueError("all mask planes have no labels (all zeros)")
        for i in range(masks.shape[0]):
            plane_max = masks[i].max()
            if plane_max == 0:
                continue
            unique = np.unique(masks[i])
            unique = unique[unique > 0]
            expected = np.arange(1, plane_max + 1)
            if not np.array_equal(unique, expected):
                raise ValueError(
                    f"mask plane '{self._masks[i]}' has non-contiguous labels "
                    f"{unique.tolist()}; labels must be 1..N with no gaps"
                )
        if np.issubdtype(image.dtype, np.integer):
            warnings.warn(
                "image has integer dtype; cp_measure expects float images "
                "normalized to [0, 1]. Consider dividing by the max value "
                "(e.g., image / 255.0 for uint8, image / 65535.0 for uint16).",
                UserWarning,
                stacklevel=2,
            )


def make_featurizer(
    channels: list[str],
    *,
    masks: list[str] | None = None,
    intensity: bool = False,
    intensity_params: dict | None = None,
    texture: bool = False,
    texture_params: dict | None = None,
    granularity: bool = False,
    granularity_params: dict | None = None,
    radial_distribution: bool = False,
    radial_distribution_params: dict | None = None,
    radial_zernikes: bool = False,
    radial_zernikes_params: dict | None = None,
    sizeshape: bool = False,
    sizeshape_params: dict | None = None,
    zernike: bool = False,
    zernike_params: dict | None = None,
    ferret: bool = False,
    correlation_pearson: bool = False,
    correlation_costes: bool = False,
    correlation_costes_params: dict | None = None,
    correlation_manders_fold: bool = False,
    correlation_manders_fold_params: dict | None = None,
    correlation_rwc: bool = False,
    correlation_rwc_params: dict | None = None,
) -> Featurizer:
    """Create a configured featurizer pipeline.

    Parameters
    ----------
    channels : list[str]
        Names for each channel in the image. The image passed to
        :meth:`Featurizer.featurize` must have ``len(channels)`` planes
        along the first axis.
    masks : list[str], optional
        Names for each mask plane. The masks array passed to
        :meth:`Featurizer.featurize` must have ``len(masks)`` planes
        along the first axis. If ``None``, defaults to ``["mask"]``.
    intensity : bool
        Compute intensity features per channel.
    intensity_params : dict, optional
        Extra keyword arguments forwarded to ``get_intensity``.
    texture : bool
        Compute texture (GLCM) features per channel.
    texture_params : dict, optional
        Extra keyword arguments forwarded to ``get_texture``
        (e.g. ``{"scale": 5, "gray_levels": 256}``).
    granularity : bool
        Compute granularity spectrum per channel.
    granularity_params : dict, optional
        Extra keyword arguments forwarded to ``get_granularity``
        (e.g. ``{"granular_spectrum_length": 8}``).
    radial_distribution : bool
        Compute radial intensity distribution per channel.
    radial_distribution_params : dict, optional
        Extra keyword arguments forwarded to ``get_radial_distribution``.
    radial_zernikes : bool
        Compute radial Zernike moments per channel.
    radial_zernikes_params : dict, optional
        Extra keyword arguments forwarded to ``get_radial_zernikes``.
    sizeshape : bool
        Compute size and shape features (mask only, run once).
    sizeshape_params : dict, optional
        Extra keyword arguments forwarded to ``get_sizeshape``.
    zernike : bool
        Compute Zernike shape features (mask only, run once).
    zernike_params : dict, optional
        Extra keyword arguments forwarded to ``get_zernike``.
    ferret : bool
        Compute Feret diameter features (mask only, run once).
    correlation_pearson : bool
        Compute Pearson correlation and slope for all channel permutations.
    correlation_costes : bool
        Compute Costes colocalization for all channel permutations.
    correlation_costes_params : dict, optional
        Extra keyword arguments forwarded to ``get_correlation_costes``.
    correlation_manders_fold : bool
        Compute Manders fold colocalization for all unique channel pairs.
    correlation_manders_fold_params : dict, optional
        Extra keyword arguments forwarded to ``get_correlation_manders_fold``.
    correlation_rwc : bool
        Compute RWC colocalization for all unique channel pairs.
    correlation_rwc_params : dict, optional
        Extra keyword arguments forwarded to ``get_correlation_rwc``.

    Returns
    -------
    Featurizer
        A configured featurizer ready for calling :meth:`~Featurizer.featurize`.

    Raises
    ------
    ValueError
        If no features are enabled or if ``channels`` is empty.
    """
    if not channels:
        raise ValueError("channels must be a non-empty list of channel names")

    if masks is None:
        masks = ["mask"]

    if not masks:
        raise ValueError("masks must be a non-empty list of mask names")

    if len(set(masks)) != len(masks):
        raise ValueError("mask names must be unique")

    # Capture all local args for lookup
    args = locals()

    # Resolve functions from bulk registries
    from cp_measure.bulk import get_core_measurements, get_correlation_measurements

    core_funcs = get_core_measurements()
    corr_funcs = get_correlation_measurements()

    channel_features: list[tuple[str, callable, dict]] = []
    for kwarg_name, registry_key in _CHANNEL_FEATURES.items():
        if args[kwarg_name]:
            params = args.get(f"{kwarg_name}_params") or {}
            channel_features.append((kwarg_name, core_funcs[registry_key], params))

    shape_features: list[tuple[str, callable, dict]] = []
    for kwarg_name, registry_key in _SHAPE_FEATURES.items():
        if args[kwarg_name]:
            params = args.get(f"{kwarg_name}_params") or {}
            shape_features.append((kwarg_name, core_funcs[registry_key], params))

    correlation_features: list[tuple[str, callable, dict, bool]] = []
    for kwarg_name, (registry_key, symmetric) in _CORRELATION_FEATURES.items():
        if args[kwarg_name]:
            params = args.get(f"{kwarg_name}_params") or {}
            correlation_features.append(
                (kwarg_name, corr_funcs[registry_key], params, symmetric)
            )

    if not channel_features and not shape_features and not correlation_features:
        raise ValueError(
            "at least one feature must be enabled "
            "(e.g., intensity=True, sizeshape=True)"
        )

    if correlation_features and len(channels) < 2:
        warnings.warn(
            "correlation features require at least 2 channels; "
            "skipping correlation since only 1 channel was provided",
            UserWarning,
            stacklevel=2,
        )
        correlation_features = []

    return Featurizer(
        channels=channels,
        masks=masks,
        channel_features=channel_features,
        shape_features=shape_features,
        correlation_features=correlation_features,
    )
