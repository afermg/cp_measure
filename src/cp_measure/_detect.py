"""Backend capability detection.

Detected ONCE at import via ``importlib.util.find_spec`` — availability is
checked without importing the package or catching ImportErrors. Dispatch reads
the flag; the resolved backend path is then called directly and unguarded
(a backend that is flagged present but raises is a real bug and must surface,
not be papered over by a try/except fallback).

Other backends (jax, ...) add their own flags here as they are wired.
"""

import importlib.util

HAS_NUMBA: bool = importlib.util.find_spec("numba") is not None
