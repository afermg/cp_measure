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
# cp_measure.numba, "faster") will be wired into _dispatch when their sibling
# packages exist; until then, selecting them raises NotImplementedError.
_REGISTRIES: dict[str, dict[str, Callable]] = {
    "core": _CORE,
    "correlation": _CORRELATION,
}

# 3D-capable subset of the core registry, by feature name.
_3D_FEATURES = ("intensity", "sizeshape", "texture", "granularity")


def _dispatch(name: str) -> dict[str, Callable]:
    from cp_measure import _ACCELERATOR

    if _ACCELERATOR is None:
        return _REGISTRIES[name]
    if _ACCELERATOR == "jax":
        raise NotImplementedError(
            f"'jax' accelerator not yet wired for {name} measurements"
        )
    if _ACCELERATOR == "numba":
        raise NotImplementedError(
            f"'numba' accelerator not yet wired for {name} measurements"
        )
    if _ACCELERATOR == "faster":
        raise NotImplementedError("'faster' logic not yet implemented")
    raise ValueError(
        f"invalid accelerator {_ACCELERATOR!r}; "
        "set via cp_measure.set_accelerator(None | 'jax' | 'numba' | 'faster')"
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
