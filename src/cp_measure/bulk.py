"""
Wrapper to fetch measurement functions in bulk. We are keeping the ugly filenames to
match the original names. This may change in the future.
"""

from typing import Callable


def get_core_measurements() -> dict[str, Callable]:
    from cp_measure.core import (
        measuregranularity,
        measureobjectintensity,
        measureobjectintensitydistribution,
        measureobjectsizeshape,
        measuretexture,
    )

    return {
        "radial_distribution": measureobjectintensitydistribution.get_radial_distribution,
        "radial_zernikes": measureobjectintensitydistribution.get_radial_zernikes,
        "intensity": measureobjectintensity.get_intensity,
        "sizeshape": measureobjectsizeshape.get_sizeshape,
        "zernike": measureobjectsizeshape.get_zernike,
        "ferret": measureobjectsizeshape.get_ferret,
        "texture": measuretexture.get_texture,
        "granularity": measuregranularity.get_granularity,
    }


def get_correlation_measurements() -> dict[str, Callable]:
    from cp_measure.core import measurecolocalization

    return {
        "costes": measurecolocalization.get_correlation_costes,
        "pearson": measurecolocalization.get_correlation_pearson,
        "manders_fold": measurecolocalization.get_correlation_manders_fold,
        "rwc": measurecolocalization.get_correlation_rwc,
    }


def get_multimask_measurements() -> dict[str, Callable]:
    from cp_measure.multimask.measureobjectneighbors import measureobjectneighbors
    from cp_measure.multimask.measureobjectoverlap import measureobjectoverlap

    return {
        "overlap": measureobjectoverlap,
        "neighbors": measureobjectneighbors,
    }
