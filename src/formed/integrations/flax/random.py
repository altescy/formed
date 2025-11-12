from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from flax import nnx

_NNX_RNGS = ContextVar[nnx.Rngs]("nnx_rngs", default=nnx.Rngs())


@contextmanager
def use_rngs(default: int | None = None) -> Iterator[nnx.Rngs]:
    """Context manager to set and restore nnx RNGs.

    This context manager allows temporarily setting the nnx RNGs
    used in Flax/nnx operations. It saves the current RNGs on entry
    and restores them on exit.

    Yields:
        The current nnx RNGs within the context.

    """
    token = _NNX_RNGS.set(nnx.Rngs(default))
    try:
        yield _NNX_RNGS.get()
    finally:
        _NNX_RNGS.reset(token)


def require_rngs() -> nnx.Rngs:
    """Get the current nnx RNGs.

    Returns:
        The current nnx RNGs.

    """
    return _NNX_RNGS.get()
