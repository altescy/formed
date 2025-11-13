"""Context management for PyTorch operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_TORCH_DEVICE = ContextVar[torch.device | None]("torch_device", default=None)


@contextmanager
def use_device(device: torch.device | str | None = None) -> Iterator[torch.device | None]:
    """Context manager to set and restore the default PyTorch device.

    This context manager allows temporarily setting the default device
    used in PyTorch operations (e.g., in ensure_torch_tensor). It saves
    the current device on entry and restores it on exit.

    Args:
        device: Device to use within the context. Can be a torch.device,
            a string like "cuda:0" or "cpu", or None.

    Yields:
        The current device within the context.

    Example:
        >>> import torch
        >>> from formed.integrations.torch import use_device, ensure_torch_tensor
        >>> import numpy as np
        >>> with use_device("cuda:0" if torch.cuda.is_available() else "cpu"):
        ...     arr = np.array([1.0, 2.0, 3.0])
        ...     tensor = ensure_torch_tensor(arr)
        ...     print(tensor.device)
        cuda:0  # or cpu if CUDA not available
    """
    device_obj = torch.device(device) if device is not None else None
    token = _TORCH_DEVICE.set(device_obj)
    try:
        yield _TORCH_DEVICE.get()
    finally:
        _TORCH_DEVICE.reset(token)


def get_device() -> torch.device | None:
    """Get the current default PyTorch device from context.

    Returns:
        The current device set in the context, or None if not set.

    Example:
        >>> from formed.integrations.torch import use_device, get_device
        >>> with use_device("cuda:0"):
        ...     print(get_device())
        cuda:0
    """
    return _TORCH_DEVICE.get()
