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

    Composes the numba implementations (``intensity``, ``granularity``) with the
    numpy implementations of every other feature — a single global "numba"
    Composes the numba implementations (``intensity`` and the ``pearson`` /
    ``manders_fold`` / ``rwc`` / ``costes`` / ``overlap`` colocalization features)
    with the numpy implementations of every other feature — a single global "numba"
    selection still yields a full, working feature set, accelerated where a
    numba backend exists. This is explicit per-function composition, NOT an
    error-driven fallback.

    Note: ``overlap`` is not in the numpy ``_CORRELATION`` registry, so the numba
    correlation registry intentionally exposes one feature the numpy one does not
    (the numba ``overlap`` backend exists and is cheap to surface). Adding
    ``overlap`` to the numpy ``_CORRELATION`` for symmetry is a separate call.
    """
    from cp_measure.core.numba import (
        get_granularity as _numba_granularity,
        get_intensity as _numba_intensity,
    Composes the numba implementations (``intensity``, ``zernike``,
    ``radial_zernikes``, ``radial_distribution``) with the numpy implementations of
    every other feature — a single global "numba" selection still yields a full,
    working feature set, accelerated where a numba backend exists. This is explicit
    per-function composition, NOT an error-driven fallback.
    """
    from cp_measure.core.numba import (
        get_intensity as _numba_intensity,
        get_radial_distribution as _numba_radial_distribution,
        get_radial_zernikes as _numba_radial_zernikes,
        get_zernike as _numba_zernike,
    )

    return {
        "core": {
            **_CORE,
            "intensity": _numba_intensity,
            "granularity": _numba_granularity,
            "zernike": _numba_zernike,
            "radial_zernikes": _numba_radial_zernikes,
            "radial_distribution": _numba_radial_distribution,
        },
        "correlation": _CORRELATION,
        get_correlation_costes as _numba_costes,
        get_correlation_manders_fold as _numba_manders_fold,
        get_correlation_overlap as _numba_overlap,
        get_correlation_pearson as _numba_pearson,
        get_correlation_rwc as _numba_rwc,
        get_intensity as _numba_intensity,
    )

    return {
        "core": {**_CORE, "intensity": _numba_intensity},
        "correlation": {
            **_CORRELATION,
            "pearson": _numba_pearson,
            "manders_fold": _numba_manders_fold,
            "rwc": _numba_rwc,
            "costes": _numba_costes,
            "overlap": _numba_overlap,
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
