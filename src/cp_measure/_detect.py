"""Backend capability detection.

Detected ONCE at import via ``importlib.util.find_spec`` — availability is
checked without importing the package or catching ImportErrors. Dispatch reads
these flags; the resolved backend path is then called directly and unguarded
(a backend that is flagged present but raises is a real bug and must surface,
not be papered over by a try/except fallback).
"""

import importlib.util

HAS_NUMBA: bool = importlib.util.find_spec("numba") is not None
HAS_JAX: bool = importlib.util.find_spec("jax") is not None


def _detect_jax_gpu() -> bool:
    if not HAS_JAX:
        return False
    import jax

    return jax.default_backend() != "cpu"


HAS_JAX_GPU: bool = _detect_jax_gpu()
