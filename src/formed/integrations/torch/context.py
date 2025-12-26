"""Context management for PyTorch operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_TORCH_DEVICE = ContextVar[torch.device | None]("torch_device", default=None)


@contextmanager
def use_device(device: str | torch.device | None = None) -> Iterator[torch.device]:
    """Context manager to set and restore the default PyTorch device.

    This context manager allows temporarily setting the default device
    used in PyTorch operations (e.g., in `ensure_torch_tensor`). It saves
    the current device on entry and restores it on exit.

    Args:
        device: Device to use within the context. Can be a torch.device,
            a string like `"cuda:0"` or `"cpu"`, or None.

    Yields:
        The current device within the context.

    Examples:
        >>> import torch
        >>> from formed.integrations.torch import use_device, ensure_torch_tensor
        >>> import numpy as np
        >>> with use_device("cuda:0" if torch.cuda.is_available() else "cpu"):
        ...     arr = np.array([1.0, 2.0, 3.0])
        ...     tensor = ensure_torch_tensor(arr)
        ...     print(tensor.device)
        cuda:0  # or cpu if CUDA not available
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if isinstance(device, str):
        device = torch.device(device)
    token = _TORCH_DEVICE.set(torch.device(device))
    try:
        yield device
    finally:
        _TORCH_DEVICE.reset(token)


def get_device() -> torch.device | None:
    """Get the current default PyTorch device from context.

    Returns:
        The current device set in the context, or `None` if not set.

    Examples:
        >>> from formed.integrations.torch import use_device, get_device
        >>> with use_device("cuda:0"):
        ...     print(get_device())
        cuda:0
    """
    return _TORCH_DEVICE.get()
