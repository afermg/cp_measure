"""
Wrapper to fetch measurement functions in bulk. We are keeping the ugly filenames to
match the original names. This may change in the future.
"""

from typing import Callable

def get_all_measurements() -> dict[str, Callable]:
    
    from cp_measure.minimal import (
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
        "granularity": measuregranularity.get_granularity,
        "texture": measuretexture.get_texture,
    }

def get_fast_measurements() -> dict[str, Callable]:
    from cp_measure.fast import (
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
