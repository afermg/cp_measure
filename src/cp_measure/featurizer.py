"""High-level featurization orchestrator for cp_measure.

Wraps all cp_measure core measurement functions into a single
:class:`Featurizer` that operates on multichannel images and
multiple named masks, producing a tidy table.

Notes
-----
Some shape features may produce NaN values for certain object
geometries. In particular, ``NormalizedMoment_0_0``,
``NormalizedMoment_0_1``, and ``NormalizedMoment_1_0`` are
mathematically undefined for objects with uniform pixel values
or degenerate shapes. These NaN values are expected and
inherent to the underlying skimage ``regionprops`` computation.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

#: Supported output formats for :meth:`Featurizer.featurize`.
ReturnFormat = Literal["pyarrow", "polars", "pandas"]


class Featurizer:
    """Configured featurization pipeline.

    Use :func:`make_featurizer` to create instances.

    Parameters
    ----------
    channels : list[str]
        Channel names corresponding to the first axis of the image array.
    masks : list[str]
        Mask names corresponding to the first axis of the masks array.
    channel_features : list[tuple[Callable, dict]]
        Per-channel features as ``(func, params)`` tuples.
    shape_features : list[tuple[Callable, dict]]
        Mask-only shape features as ``(func, params)`` tuples.
    correlation_features : list[tuple[Callable, dict, bool]]
        Pairwise correlation features as ``(func, params, symmetric)``
        tuples. Symmetric metrics use combinations; asymmetric use
        permutations.
    """

    def __init__(
        self,
        channels: list[str],
        masks: list[str],
        channel_features: list[tuple[Callable, dict]],
        shape_features: list[tuple[Callable, dict]],
        correlation_features: list[tuple[Callable, dict, bool]],
    ):
        self._channels = list(channels)
        self._masks = list(masks)
        self._channel_features = channel_features
        self._shape_features = shape_features
        self._correlation_features = correlation_features

    def featurize(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        *,
        return_as: ReturnFormat = "pyarrow",
    ) -> pa.Table | pl.DataFrame | pd.DataFrame:
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
        return_as : {"pyarrow", "polars", "pandas"}, default "pyarrow"
            Output format. ``"pyarrow"`` returns a :class:`pyarrow.Table`,
            ``"polars"`` returns a :class:`polars.DataFrame`, and
            ``"pandas"`` returns a :class:`pandas.DataFrame`.
            The requested library must be importable; a helpful
            :class:`ImportError` is raised otherwise.

        Returns
        -------
        pyarrow.Table | polars.DataFrame | pandas.DataFrame
            Table with one row per labeled object (union of all labels
            across masks). The ``"label"`` column contains the label ID.
            Feature columns are named ``{mask}_{feature}_{channel}`` for
            per-channel features, ``{mask}_{feature}`` for shape features,
            and ``{mask}_{feature}_{ch1}_{ch2}`` for correlation features.
            Labels missing from a given mask have null for that mask's columns.
            Rows are sorted by ``label`` in ascending order.
        """
        _check_return_format(return_as)
        self._validate(image, masks)

        # Shape features (sizeshape, zernike, ferret) are purely geometric
        # and do not depend on pixel values. Their function signatures
        # require a pixels argument, but it is ignored; pass zeros.
        _dummy_pixels = np.zeros(image.shape[1:], dtype=image.dtype)

        per_mask_tables: list[pa.Table] = []

        for mask_idx, mask_name in enumerate(self._masks):
            mask_2d = masks[mask_idx]
            # n_labels == mask_2d.max() is safe because _validate enforces
            # contiguous labels 1..N with no gaps.
            n_labels = mask_2d.max()

            if n_labels == 0:
                continue

            results: dict[str, np.ndarray | list] = {}

            # Shape features — purely geometric, run once per mask
            for func, params in self._shape_features:
                for key, values in func(mask_2d, _dummy_pixels, **params).items():
                    results[f"{mask_name}_{key}"] = values

            # Per-channel features
            for ch_idx, ch_name in enumerate(self._channels):
                pixels = image[ch_idx]
                for func, params in self._channel_features:
                    for key, values in func(mask_2d, pixels, **params).items():
                        results[f"{mask_name}_{key}_{ch_name}"] = values

            # Correlation features — symmetric use combinations, asymmetric
            # use permutations to capture both channel orderings
            n_ch = len(self._channels)
            for func, params, symmetric in self._correlation_features:
                pairs = (
                    itertools.combinations(range(n_ch), 2)
                    if symmetric
                    else itertools.permutations(range(n_ch), 2)
                )
                for ch_i, ch_j in pairs:
                    result = func(
                        pixels_1=image[ch_i],
                        pixels_2=image[ch_j],
                        masks=mask_2d,
                        **params,
                    )
                    for key, values in result.items():
                        col = f"{mask_name}_{key}_{self._channels[ch_i]}_{self._channels[ch_j]}"
                        results[col] = values

            labels = np.arange(1, n_labels + 1)
            arrays = [pa.array(labels)]
            names = ["label"]
            for col_name, col_values in results.items():
                arrays.append(pa.array(col_values))
                names.append(col_name)
            mask_table = pa.table(arrays, names=names)
            per_mask_tables.append(mask_table)

        # Full outer join across masks on the "label" column using
        # vectorized pc.take (C++ kernel) instead of per-cell .as_py().
        table = per_mask_tables[0]
        for other in per_mask_tables[1:]:
            labels_left = table.column("label").to_pylist()
            labels_right = other.column("label").to_pylist()
            all_labels = sorted(set(labels_left) | set(labels_right))

            left_idx = {lab: i for i, lab in enumerate(labels_left)}
            right_idx = {lab: i for i, lab in enumerate(labels_right)}

            # Index arrays with None for missing labels -> null in take output
            left_indices = pa.array(
                [left_idx.get(lab) for lab in all_labels], type=pa.int64()
            )
            right_indices = pa.array(
                [right_idx.get(lab) for lab in all_labels], type=pa.int64()
            )

            merged_arrays: list[pa.Array] = [pa.array(all_labels, type=pa.int64())]
            merged_names: list[str] = ["label"]

            for col_name in table.column_names:
                if col_name == "label":
                    continue
                merged_arrays.append(pc.take(table.column(col_name), left_indices))
                merged_names.append(col_name)

            for col_name in other.column_names:
                if col_name == "label":
                    continue
                merged_arrays.append(pc.take(other.column(col_name), right_indices))
                merged_names.append(col_name)

            table = pa.table(merged_arrays, names=merged_names)

        # Sort by label
        table = table.sort_by("label")

        if return_as == "pyarrow":
            return table
        if return_as == "polars":
            import polars as pl

            return pl.from_arrow(table)
        # return_as == "pandas" — availability already checked by _check_return_format
        return table.to_pandas()

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
        if masks.min() < 0:
            raise ValueError("masks must not contain negative labels")
        all_empty = True
        for i in range(masks.shape[0]):
            plane_max = masks[i].max()
            if plane_max == 0:
                continue
            all_empty = False
            counts = np.bincount(masks[i].ravel())
            # Labels 1..plane_max must all be present (non-zero count)
            if len(counts) <= plane_max or np.any(counts[1 : plane_max + 1] == 0):
                unique = np.unique(masks[i])
                unique = unique[unique > 0]
                raise ValueError(
                    f"mask plane '{self._masks[i]}' has non-contiguous labels "
                    f"{unique.tolist()}; labels must be 1..N with no gaps"
                )
        if all_empty:
            raise ValueError("all mask planes have no labels (all zeros)")
        if np.issubdtype(image.dtype, np.integer):
            warnings.warn(
                "image has integer dtype; cp_measure expects float images "
                "normalized to [0, 1]. Consider dividing by the max value "
                "(e.g., image / 255.0 for uint8, image / 65535.0 for uint16).",
                UserWarning,
                stacklevel=2,
            )


def _check_return_format(return_as: ReturnFormat) -> None:
    """Validate return format and check library availability early."""
    import importlib.util

    valid = ("pyarrow", "polars", "pandas")
    if return_as not in valid:
        raise ValueError(f"return_as must be one of {valid!r}, got {return_as!r}")
    if return_as == "polars" and importlib.util.find_spec("polars") is None:
        raise ImportError(
            "return_as='polars' requires the polars package. "
            "Install it with: pip install polars"
        )
    if return_as == "pandas" and importlib.util.find_spec("pandas") is None:
        raise ImportError(
            "return_as='pandas' requires the pandas package. "
            "Install it with: pip install pandas"
        )


def make_featurizer(
    channels: list[str],
    *,
    masks: list[str] | None = None,
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
    ferret: bool = True,
    correlation_pearson: bool = True,
    correlation_costes: bool = True,
    correlation_costes_params: dict | None = None,
    correlation_manders_fold: bool = True,
    correlation_manders_fold_params: dict | None = None,
    correlation_rwc: bool = True,
    correlation_rwc_params: dict | None = None,
) -> Featurizer:
    """Create a configured featurizer pipeline.

    By default all features are enabled. Pass ``feature=False`` to
    disable individual features.

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
        Compute size and shape features (purely geometric, run once
        per mask). All sizeshape features depend only on the mask
        geometry; columns have no channel suffix.
    sizeshape_params : dict, optional
        Extra keyword arguments forwarded to ``get_sizeshape``.
    zernike : bool
        Compute Zernike shape features (mask only, run once).
    zernike_params : dict, optional
        Extra keyword arguments forwarded to ``get_zernike``.
    ferret : bool
        Compute Feret diameter features (mask only, run once).
        ``get_ferret`` accepts no extra parameters.
    correlation_pearson : bool
        Compute Pearson correlation and slope for all channel permutations.
        ``get_correlation_pearson`` accepts no extra parameters.
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
        If no features are enabled, if ``channels`` or ``masks`` is empty,
        if channel names are not unique, or if mask names are not unique.
    """
    if not channels:
        raise ValueError("channels must be a non-empty list of channel names")

    if len(set(channels)) != len(channels):
        raise ValueError("channel names must be unique")

    if masks is None:
        masks = ["mask"]

    if not masks:
        raise ValueError("masks must be a non-empty list of mask names")

    if len(set(masks)) != len(masks):
        raise ValueError("mask names must be unique")

    from cp_measure.bulk import get_core_measurements, get_correlation_measurements

    core_funcs = get_core_measurements()
    corr_funcs = get_correlation_measurements()

    channel_features: list[tuple[Callable, dict]] = []
    if intensity:
        channel_features.append((core_funcs["intensity"], intensity_params or {}))
    if texture:
        channel_features.append((core_funcs["texture"], texture_params or {}))
    if granularity:
        channel_features.append((core_funcs["granularity"], granularity_params or {}))
    if radial_distribution:
        channel_features.append(
            (core_funcs["radial_distribution"], radial_distribution_params or {})
        )
    if radial_zernikes:
        channel_features.append(
            (core_funcs["radial_zernikes"], radial_zernikes_params or {})
        )

    # Shape features are purely geometric (do not depend on pixel values).
    shape_features: list[tuple[Callable, dict]] = []
    if sizeshape:
        shape_features.append((core_funcs["sizeshape"], sizeshape_params or {}))
    if zernike:
        shape_features.append((core_funcs["zernike"], zernike_params or {}))
    if ferret:
        shape_features.append((core_funcs["ferret"], {}))

    # Symmetric metrics produce identical results for (A, B) and (B, A)
    # — either directly or via _1/_2 suffix swap — so combinations suffice.
    # Asymmetric metrics must be run for both orderings (permutations).
    correlation_features: list[tuple[Callable, dict, bool]] = []
    if correlation_pearson:
        correlation_features.append((corr_funcs["pearson"], {}, False))
    if correlation_costes:
        correlation_features.append(
            (corr_funcs["costes"], correlation_costes_params or {}, False)
        )
    if correlation_manders_fold:
        correlation_features.append(
            (corr_funcs["manders_fold"], correlation_manders_fold_params or {}, True)
        )
    if correlation_rwc:
        correlation_features.append(
            (corr_funcs["rwc"], correlation_rwc_params or {}, True)
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
