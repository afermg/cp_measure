"""
Wrapper to fetch measurement functions in bulk. We are keeping the ugly filenames to
match the original names. This may change in the future.
"""

from typing import Callable

from cp_measure.core import (
    measurecolocalization,
    measuregranularity,
    measureobjectintensity,
    measureobjectintensitydistribution,
    measureobjectsizeshape,
    measuretexture,
)


_CORE: dict[str, Callable] = {
    "radial_distribution": measureobjectintensitydistribution.get_radial_distribution,
    "radial_zernikes": measureobjectintensitydistribution.get_radial_zernikes,
    "intensity": measureobjectintensity.get_intensity,
    "sizeshape": measureobjectsizeshape.get_sizeshape,
    "zernike": measureobjectsizeshape.get_zernike,
    "feret": measureobjectsizeshape.get_feret,
    "texture": measuretexture.get_texture,
    "granularity": measuregranularity.get_granularity,
}

_CORRELATION: dict[str, Callable] = {
    "costes": measurecolocalization.get_correlation_costes,
    "pearson": measurecolocalization.get_correlation_pearson,
    "manders_fold": measurecolocalization.get_correlation_manders_fold,
    "rwc": measurecolocalization.get_correlation_rwc,
}

# Default (None accelerator) registries. Optional backends (cp_measure.jax,
# cp_measure.numba, "fastest") will be wired into _dispatch when their sibling
# packages exist; until then, selecting them raises NotImplementedError.
_REGISTRIES: dict[str, dict[str, Callable]] = {
    "core": _CORE,
    "correlation": _CORRELATION,
}

# 3D-capable subset of the core registry, by feature name.
_3D_FEATURES = ("intensity", "sizeshape", "texture", "granularity")


def _numba_registries() -> dict[str, dict[str, Callable]]:
    """Registries for the 'numba' accelerator.

    Composes the numba implementations (intensity, granularity, zernike, radial_zernikes,
    radial_distribution, texture, feret, sizeshape, and the colocalization features including
    costes and overlap) with the numpy implementations of every other feature, so a single
    global "numba" selection yields a full, accelerated-where-available feature set. This is
    explicit per-function composition, NOT an error-driven fallback. ``overlap`` is numba-only
    (not present in the numpy ``_CORRELATION`` registry).
    """
    from cp_measure.core.numba import (
        get_correlation_costes,
        get_correlation_manders_fold,
        get_correlation_overlap,
        get_correlation_pearson,
        get_correlation_rwc,
        get_feret,
        get_granularity,
        get_intensity,
        get_radial_distribution,
        get_radial_zernikes,
        get_sizeshape,
        get_texture,
        get_zernike,
    )

    return {
        "core": {
            **_CORE,
            "intensity": get_intensity,
            "granularity": get_granularity,
            "zernike": get_zernike,
            "radial_zernikes": get_radial_zernikes,
            "radial_distribution": get_radial_distribution,
            "texture": get_texture,
            "feret": get_feret,
            "sizeshape": get_sizeshape,
        },
        "correlation": {
            **_CORRELATION,
            "pearson": get_correlation_pearson,
            "manders_fold": get_correlation_manders_fold,
            "rwc": get_correlation_rwc,
            "costes": get_correlation_costes,
            "overlap": get_correlation_overlap,
        },
    }


def _dispatch(name: str) -> dict[str, Callable]:
    from cp_measure import _ACCELERATOR

    if _ACCELERATOR is None:
        return _REGISTRIES[name]
    if _ACCELERATOR == "jax":
        raise NotImplementedError(
            f"'jax' accelerator not yet wired for {name} measurements"
        )
    if _ACCELERATOR == "numba":
        from cp_measure._detect import HAS_NUMBA

        if not HAS_NUMBA:
            raise RuntimeError(
                "accelerator 'numba' selected but numba is not installed; "
                "you can install it via `pip install cp_measure[numba]`"
            )
        return _numba_registries()[name]
    if _ACCELERATOR == "fastest":
        raise NotImplementedError("'fastest' logic not yet implemented")
    raise ValueError(
        f"invalid accelerator {_ACCELERATOR!r}; "
        "set via cp_measure.set_accelerator(None | 'jax' | 'numba' | 'fastest')"
    )


def get_core_measurements() -> dict[str, Callable]:
    return _dispatch("core")


def get_core_measurements_3d() -> dict[str, Callable]:
    """Return only measurements that support 3D input."""
    core = _dispatch("core")
    return {k: core[k] for k in _3D_FEATURES}


def get_correlation_measurements() -> dict[str, Callable]:
    return _dispatch("correlation")


def get_multimask_measurements() -> dict[str, Callable]:
    from cp_measure.multimask.measureobjectneighbors import measureobjectneighbors
    from cp_measure.multimask.measureobjectoverlap import measureobjectoverlap

    return {
        "overlap": measureobjectoverlap,
        "neighbors": measureobjectneighbors,
    }
