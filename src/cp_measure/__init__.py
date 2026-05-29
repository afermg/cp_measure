_VALID_ACCELERATORS = (None, "jax", "numba", "faster")
_ACCELERATOR: str | None = None


def set_accelerator(backend: str | None) -> None:
    if backend not in _VALID_ACCELERATORS:
        raise ValueError(
            f"unknown accelerator {backend!r}; expected one of {_VALID_ACCELERATORS}"
        )
    global _ACCELERATOR
    _ACCELERATOR = backend
