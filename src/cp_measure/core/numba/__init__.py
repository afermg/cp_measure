"""Numba-accelerated backend.

Selected globally via ``cp_measure.set_accelerator("numba")``, or by importing an
implementation directly (``from cp_measure.core.numba.measuretexture import
get_texture``). Requires the optional ``numba`` extra; availability is gated by
``cp_measure._detect.HAS_NUMBA``.

One module per numpy module it accelerates, same name:

- ``measureobjectintensity`` -> ``get_intensity``
- ``measuretexture`` -> ``get_texture``

The global "numba" accelerator composes these with the numpy implementations of
every other feature (see ``cp_measure.bulk``). Nothing is re-exported here: each
implementation pulls in its own dependencies, and importing one should not drag
in the rest.
"""
