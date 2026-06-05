"""
Wrapper to fetch measurement functions in bulk. We are keeping the ugly filenames to
match the original names. This may change in the future.
"""

from functools import partial
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

    Composes the numba implementations (currently ``intensity`` only) with the
    numpy implementations of every other feature — a single global "numba"
    selection still yields a full, working feature set, accelerated where a
    numba backend exists. This is explicit per-function composition, NOT an
    error-driven fallback.
    """
    from cp_measure.core.numba import get_intensity as _numba_intensity

    return {
        "core": {**_CORE, "intensity": _numba_intensity},
        "correlation": _CORRELATION,
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


# Features whose result honours the `legacy` percentile convention (see
# cp_measure.core.measureobjectintensity.get_intensity). Extend as more lanes
# gain a legacy/new split.
_LEGACY_FEATURES = ("intensity",)


def _apply_legacy(core: dict[str, Callable], legacy: bool) -> dict[str, Callable]:
    """Bind ``legacy=True`` into the features that honour it; no-op when False."""
    if not legacy:
        return core
    return {
        name: (partial(fn, legacy=True) if name in _LEGACY_FEATURES else fn)
        for name, fn in core.items()
    }


def get_core_measurements(legacy: bool = False) -> dict[str, Callable]:
    """Core per-object measurement functions.

    ``legacy`` (default False) selects the original CellProfiler ``n*q`` percentile
    convention for the intensity quartile/MAD features instead of the default
    ``numpy.percentile`` one; see
    :func:`cp_measure.core.measureobjectintensity.get_intensity`.
    """
    return _apply_legacy(_dispatch("core"), legacy)


def get_core_measurements_3d(legacy: bool = False) -> dict[str, Callable]:
    """Return only measurements that support 3D input (see ``legacy`` above)."""
    core = {k: _dispatch("core")[k] for k in _3D_FEATURES}
    return _apply_legacy(core, legacy)


def get_correlation_measurements() -> dict[str, Callable]:
    return _dispatch("correlation")


def get_multimask_measurements() -> dict[str, Callable]:
    from cp_measure.multimask.measureobjectneighbors import measureobjectneighbors
    from cp_measure.multimask.measureobjectoverlap import measureobjectoverlap

    return {
        "overlap": measureobjectoverlap,
        "neighbors": measureobjectneighbors,
    }
