"""High-level featurization orchestrator for cp_measure.

Wraps all cp_measure core measurement functions into a single
:class:`Featurizer` that operates on multichannel images and
multiple named object masks, producing a tidy table.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


class Featurizer:
    """Configured featurization pipeline.

    Use :func:`make_featurizer` to create instances.

    Parameters
    ----------
    channels : list[str]
        Channel names corresponding to the first axis of the image array.
    objects : list[str]
        Object names corresponding to the first axis of the masks array.
    channel_features : list[tuple[Callable, dict, bool]]
        Per-channel features as ``(func, params, volumetric)`` tuples.
        *volumetric* is ``True`` when the function supports 3D spatial
        data; ``False`` for 2D-only functions.
    shape_features : list[tuple[Callable, dict, bool]]
        Mask-only shape features as ``(func, params, volumetric)`` tuples.
    correlation_features : list[tuple[Callable, dict, bool]]
        Pairwise correlation features as ``(func, params, symmetric)``
        tuples. Symmetric metrics use combinations; asymmetric use
        permutations. All correlation functions support volumetric data.
    """

    def __init__(
        self,
        channels: list[str],
        objects: list[str],
        channel_features: list[tuple[Callable, dict, bool]],
        shape_features: list[tuple[Callable, dict, bool]],
        correlation_features: list[tuple[Callable, dict, bool]],
    ):
        self._channels = list(channels)
        self._objects = list(objects)
        self._channel_features = channel_features
        self._shape_features = shape_features
        self._correlation_features = correlation_features

    def featurize(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        *,
        return_as: Literal["pyarrow", "polars", "pandas", "dict"] = "pyarrow",
    ) -> pa.Table | pl.DataFrame | pd.DataFrame | dict:
        """Compute all configured features for the given image and masks.

        Parameters
        ----------
        image : numpy.ndarray
            Multichannel image with shape ``(C, H, W)`` for 2D spatial data
            or ``(C, Z, H, W)`` for 3D volumetric data. ``C`` must match
            the number of channel names provided at construction time.
        masks : numpy.ndarray
            Integer-labeled masks with shape ``(M, H, W)`` or
            ``(M, Z, H, W)``. Must have the same ``ndim`` as *image*.
            ``M`` must match the number of object names provided at
            construction time. Background is 0, each object has a unique
            positive integer label per object mask. For volumetric (4D) inputs,
            2D-only features (radial_distribution, radial_zernikes, zernike,
            ferret) are automatically skipped with a warning.
        return_as : {"pyarrow", "polars", "pandas", "dict"}, default "pyarrow"
            Output format. ``"pyarrow"`` returns a :class:`pyarrow.Table`,
            ``"polars"`` returns a :class:`polars.DataFrame`,
            ``"pandas"`` returns a :class:`pandas.DataFrame`, and
            ``"dict"`` returns a plain Python dictionary of lists
            (via :meth:`pyarrow.Table.to_pydict`).
            The requested library must be importable; a helpful
            :class:`ImportError` is raised otherwise.

        Returns
        -------
        pyarrow.Table | polars.DataFrame | pandas.DataFrame | dict
            Table with one row per labeled object (union of all labels
            across masks). The ``"label"`` column contains the label ID.
            Feature columns are named ``{object}__{feature}__{channel}``
            for per-channel features, ``{object}__{feature}`` for shape
            features, and ``{object}__{feature}__{ch1}__{ch2}`` for
            correlation features. Labels missing from a given mask have
            null for that mask's columns.
        """
        _check_return_format(return_as)
        self._validate(image, masks)

        is_volumetric = image.ndim == 4

        if is_volumetric:
            skipped_names = [
                f.__name__
                for f, _p, v in (*self._channel_features, *self._shape_features)
                if not v
            ]
            channel_feats = [(f, p) for f, p, v in self._channel_features if v]
            shape_feats = [(f, p) for f, p, v in self._shape_features if v]

            if not channel_feats and not shape_feats and not self._correlation_features:
                raise ValueError(
                    "no features left for volumetric input — all enabled "
                    f"features are 2D-only ({', '.join(skipped_names)})"
                )
            if skipped_names:
                warnings.warn(
                    f"Skipped {len(skipped_names)} 2D-only feature(s) for "
                    f"volumetric input ({', '.join(skipped_names)})",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            channel_feats = [(f, p) for f, p, _v in self._channel_features]
            shape_feats = [(f, p) for f, p, _v in self._shape_features]

        # Shape features (sizeshape, zernike, ferret) are purely geometric
        # and do not depend on pixel values. Their function signatures
        # require a pixels argument, but it is ignored; pass zeros.
        _dummy_pixels = np.zeros(image.shape[1:], dtype=image.dtype)

        per_object_tables: list[pa.Table] = []

        for mask_idx, object_name in enumerate(self._objects):
            mask = masks[mask_idx]
            # n_labels == mask.max() is safe because _validate enforces
            # contiguous labels 1..N with no gaps.
            n_labels = mask.max()

            if n_labels == 0:
                continue

            results: dict[str, np.ndarray | list] = {}

            # Shape features — purely geometric, run once per mask
            for func, params in shape_feats:
                for key, values in func(mask, _dummy_pixels, **params).items():
                    results[f"{object_name}__{key}"] = values

            # Per-channel features
            for ch_idx, ch_name in enumerate(self._channels):
                pixels = image[ch_idx]
                for func, params in channel_feats:
                    for key, values in func(mask, pixels, **params).items():
                        results[f"{object_name}__{key}__{ch_name}"] = values

            # Correlation features — symmetric use combinations, asymmetric
            # use permutations to capture both channel orderings
            n_ch = len(self._channels)
            for func, params, symmetric in self._correlation_features:
                iter_fn = (
                    itertools.combinations if symmetric else itertools.permutations
                )
                for ch_i, ch_j in iter_fn(range(n_ch), 2):
                    result = func(
                        pixels_1=image[ch_i],
                        pixels_2=image[ch_j],
                        masks=mask,
                        **params,
                    )
                    for key, values in result.items():
                        col = f"{object_name}__{key}__{self._channels[ch_i]}__{self._channels[ch_j]}"
                        results[col] = values

            labels = np.arange(1, n_labels + 1, dtype=np.int32)
            arrays = [pa.array(labels)]
            names = ["label"]
            for col_name, col_values in results.items():
                arrays.append(pa.array(col_values))
                names.append(col_name)
            mask_table = pa.table(arrays, names=names)
            per_object_tables.append(mask_table)

        # Full outer join across masks on the "label" column
        table = per_object_tables[0]
        for other in per_object_tables[1:]:
            table = table.join(other, keys="label", join_type="full outer")

        if return_as == "pyarrow":
            return table
        if return_as == "polars":
            import polars as pl

            return pl.from_arrow(table)
        if return_as == "pandas":
            return table.to_pandas()
        # return_as == "dict"
        return table.to_pydict()

    def _validate(self, image: np.ndarray, masks: np.ndarray) -> None:
        """Validate inputs before featurization."""
        if image.ndim not in (3, 4):
            raise ValueError(
                f"image must be 3D (C, H, W) or 4D (C, Z, H, W), "
                f"got shape {image.shape}"
            )
        if masks.ndim not in (3, 4):
            raise ValueError(
                f"masks must be 3D (M, H, W) or 4D (M, Z, H, W), "
                f"got shape {masks.shape}"
            )
        if image.ndim != masks.ndim:
            raise ValueError(
                f"image and masks must have the same number of dimensions, "
                f"got image.ndim={image.ndim} and masks.ndim={masks.ndim}"
            )
        if image.shape[0] != len(self._channels):
            raise ValueError(
                f"image has {image.shape[0]} channels but "
                f"{len(self._channels)} channel names were provided"
            )
        if masks.shape[0] != len(self._objects):
            raise ValueError(
                f"masks has {masks.shape[0]} object masks but "
                f"{len(self._objects)} object names were provided"
            )
        if image.shape[1:] != masks.shape[1:]:
            raise ValueError(
                f"spatial dims mismatch: image {image.shape[1:]}, "
                f"masks {masks.shape[1:]}"
            )
        if not np.issubdtype(masks.dtype, np.integer):
            raise TypeError(f"masks must be integer dtype, got {masks.dtype}")
        all_empty = True
        for i in range(masks.shape[0]):
            mask_max = masks[i].max()
            if mask_max == 0:
                continue
            all_empty = False
            counts = np.bincount(masks[i].ravel())
            # Labels 1..mask_max must all be present (non-zero count)
            if len(counts) <= mask_max or np.any(counts[1 : mask_max + 1] == 0):
                unique = np.unique(masks[i])
                unique = unique[unique > 0]
                raise ValueError(
                    f"mask '{self._objects[i]}' has non-contiguous labels "
                    f"{unique.tolist()}; cp_measure requires labels to be "
                    f"1..N with no gaps"
                )
        if all_empty:
            raise ValueError("all masks have no labels (all zeros)")
        if np.issubdtype(image.dtype, np.integer):
            warnings.warn(
                "image has integer dtype; cp_measure expects float images "
                "normalized to [0, 1]. Consider dividing by the max value "
                "(e.g., image / 255.0 for uint8, image / 65535.0 for uint16).",
                UserWarning,
                stacklevel=2,
            )


def _check_return_format(
    return_as: Literal["pyarrow", "polars", "pandas", "dict"],
) -> None:
    """Validate return format and check library availability early."""
    import importlib.util

    valid = ("pyarrow", "polars", "pandas", "dict")
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
        :meth:`Featurizer.featurize` must have ``len(channels)`` channels
        along the first axis.
    objects : list[str], optional
        Names for each object mask. The masks array passed to
        :meth:`Featurizer.featurize` must have ``len(objects)`` object
        masks along the first axis. If ``None``, defaults to ``["object"]``.
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
        **2D-only** — automatically skipped for volumetric (4D) inputs.
    radial_distribution_params : dict, optional
        Extra keyword arguments forwarded to ``get_radial_distribution``.
    radial_zernikes : bool
        Compute radial Zernike moments per channel.
        **2D-only** — automatically skipped for volumetric (4D) inputs.
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
        **2D-only** — automatically skipped for volumetric (4D) inputs.
    zernike_params : dict, optional
        Extra keyword arguments forwarded to ``get_zernike``.
    ferret : bool
        Compute Feret diameter features (mask only, run once).
        **2D-only** — automatically skipped for volumetric (4D) inputs.
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
        If no features are enabled, if ``channels`` or ``objects`` is empty,
        if channel names are not unique, or if object names are not unique.

    Examples
    --------
    >>> from cp_measure.featurizer import make_featurizer
    >>> f = make_featurizer(["DNA", "ER"], objects=["nuclei", "cells"])
    >>> table = f.featurize(image, masks)  # (C, H, W) / (C, Z, H, W)
    """
    if not channels:
        raise ValueError("channels must be a non-empty list of channel names")

    if len(set(channels)) != len(channels):
        raise ValueError("channel names must be unique")

    if objects is None:
        objects = ["object"]

    if not objects:
        raise ValueError("objects must be a non-empty list of object names")

    if len(set(objects)) != len(objects):
        raise ValueError("object names must be unique")

    from cp_measure.bulk import get_core_measurements, get_correlation_measurements

    core_funcs = get_core_measurements()
    corr_funcs = get_correlation_measurements()

    # The third element of each tuple indicates volumetric (3D spatial)
    # support.  2D-only features are automatically skipped for 4D inputs.
    channel_features: list[tuple[Callable, dict, bool]] = []
    if intensity:
        channel_features.append((core_funcs["intensity"], intensity_params or {}, True))
    if texture:
        channel_features.append((core_funcs["texture"], texture_params or {}, True))
    if granularity:
        channel_features.append(
            (core_funcs["granularity"], granularity_params or {}, True)
        )
    if radial_distribution:
        channel_features.append(
            (core_funcs["radial_distribution"], radial_distribution_params or {}, False)
        )
    if radial_zernikes:
        channel_features.append(
            (core_funcs["radial_zernikes"], radial_zernikes_params or {}, False)
        )

    # Shape features are purely geometric (do not depend on pixel values).
    shape_features: list[tuple[Callable, dict, bool]] = []
    if sizeshape:
        shape_features.append((core_funcs["sizeshape"], sizeshape_params or {}, True))
    if zernike:
        shape_features.append((core_funcs["zernike"], zernike_params or {}, False))
    if ferret:
        shape_features.append((core_funcs["ferret"], {}, False))

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
        objects=objects,
        channel_features=channel_features,
        shape_features=shape_features,
        correlation_features=correlation_features,
    )
