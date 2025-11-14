"""Utility functions for PyTorch integration."""

from collections.abc import Callable, Sequence
from typing import Literal

import torch
import torch.nn.functional as F

from .context import get_device
from .types import TensorCompatible


def ensure_torch_tensor(
    x: TensorCompatible,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert array-like objects to PyTorch tensors.

    This function converts various array-like objects (numpy arrays, lists, etc.)
    to PyTorch tensors. If the input is already a tensor, it returns it with the
    appropriate dtype and device.

    The device can be specified explicitly via the `device` parameter, or it will
    be taken from the context set by `use_device()`. If neither is provided and
    the input is not already a tensor, the tensor will be created on CPU.

    Args:
        x: Input data (tensor, numpy array, list, etc.)
        dtype: Optional dtype for the output tensor.
        device: Optional device for the output tensor. If None, uses the device
            from context (set by use_device()). If the input is already a tensor,
            its device is preserved unless explicitly specified.

    Returns:
        PyTorch tensor on the specified device with the specified dtype.

    Example:
        >>> import numpy as np
        >>> from formed.integrations.torch import ensure_torch_tensor, use_device
        >>> arr = np.array([1, 2, 3])
        >>>
        >>> # Without context
        >>> tensor = ensure_torch_tensor(arr)
        >>> tensor.device
        device(type='cpu')
        >>>
        >>> # With context
        >>> with use_device("cuda:0"):
        ...     tensor = ensure_torch_tensor(arr)
        ...     print(tensor.device)
        cuda:0
    """
    # Determine target device
    if device is None:
        device = get_device()

    if isinstance(x, torch.Tensor):
        # If already a tensor, convert dtype/device as needed
        needs_dtype_conversion = dtype is not None and x.dtype != dtype
        needs_device_conversion = device is not None and x.device != torch.device(device)

        if needs_dtype_conversion or needs_device_conversion:
            kwargs = {}
            if dtype is not None:
                kwargs["dtype"] = dtype
            if device is not None:
                kwargs["device"] = device
            return x.to(**kwargs)
        return x

    # Convert numpy arrays, handling float64 -> float32 conversion
    import numpy as np

    if isinstance(x, np.ndarray):
        if dtype is None and x.dtype == np.float64:
            # Default: convert float64 to float32 for PyTorch
            dtype = torch.float32
        tensor = torch.from_numpy(x)
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    # Convert other array-like objects
    tensor = torch.as_tensor(x)
    if dtype is not None:
        tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def determine_ndim(
    first: int,
    *args: int | Callable[[int], int] | None,
) -> int:
    output_dim = first
    for arg in args:
        if arg is None:
            continue
        if callable(arg):
            output_dim = arg(output_dim)
        else:
            output_dim = arg
    return output_dim


PoolingMethod = Literal["mean", "max", "min", "sum", "first", "last"]


def masked_pool(
    inputs: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    pooling: PoolingMethod | Sequence[PoolingMethod] = "mean",
    normalize: bool = False,
) -> torch.Tensor:
    """Apply masked pooling over the sequence dimension.

    Args:
        inputs: Input tensor of shape (batch_size, seq_len, feature_dim).
        mask: Mask tensor of shape (batch_size, seq_len). True/1 indicates valid positions.
        pooling: Pooling method or sequence of methods.
        normalize: Whether to L2-normalize before pooling.

    Returns:
        Pooled tensor of shape (batch_size, feature_dim * num_pooling_methods).

    """
    if normalize:
        inputs = F.normalize(inputs, p=2, dim=-1)

    if mask is None:
        mask = torch.ones(inputs.shape[:-1], dtype=torch.bool, device=inputs.device)

    # Convert mask to boolean if needed
    if mask.dtype != torch.bool:
        mask = mask.bool()

    pooling_methods = [pooling] if isinstance(pooling, str) else list(pooling)
    results = []

    for method in pooling_methods:
        if method == "mean":
            # Masked mean
            masked_inputs = inputs * mask.unsqueeze(-1)
            pooled = masked_inputs.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        elif method == "max":
            # Masked max
            masked_inputs = inputs.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled, _ = masked_inputs.max(dim=1)
        elif method == "min":
            # Masked min
            masked_inputs = inputs.masked_fill(~mask.unsqueeze(-1), float("inf"))
            pooled, _ = masked_inputs.min(dim=1)
        elif method == "sum":
            # Masked sum
            masked_inputs = inputs * mask.unsqueeze(-1)
            pooled = masked_inputs.sum(dim=1)
        elif method == "first":
            # First token
            pooled = inputs[:, 0, :]
        elif method == "last":
            # Last valid token
            # Find the index of the last valid token for each sequence
            lengths = mask.sum(dim=1).clamp(min=1) - 1  # -1 because indices are 0-based
            batch_indices = torch.arange(inputs.size(0), device=inputs.device)
            pooled = inputs[batch_indices, lengths.long()]
        else:
            raise ValueError(f"Unknown pooling method: {method}")

        results.append(pooled)

    return torch.cat(results, dim=-1) if len(results) > 1 else results[0]
