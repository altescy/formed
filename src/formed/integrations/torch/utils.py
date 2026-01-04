"""Utility functions for PyTorch integration."""

import random
from collections.abc import Callable, Sequence
from typing import Literal, Optional, Union, cast

import numpy
import torch
import torch.nn.functional as F

from .context import get_device
from .types import ModelInputT, TensorCompatible


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility across torch, numpy, and random.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_torch_tensor(
    x: TensorCompatible,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
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
            from context (set by `use_device()`). If the input is already a tensor,
            its device is preserved unless explicitly specified.

    Returns:
        PyTorch tensor on the specified device with the specified dtype.

    Examples:
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


def move_to_device(inputs: ModelInputT, device: Optional[Union[torch.device, str]]) -> ModelInputT:
    """Move tensor inputs to the appropriate device.

    This function only moves existing torch.Tensor objects to the target device.
    Other types (numpy arrays, primitives, etc.) are left unchanged.
    Users should explicitly convert numpy arrays to tensors in their model's
    forward method using `ensure_torch_tensor()`.
    """
    from typing import Any

    visited: set[int] = set()

    def _move(obj: Any) -> Any:
        # Handle tensors - move to device
        if isinstance(obj, torch.Tensor):
            return obj.to(device)

        # Handle primitives and None - no conversion needed
        if obj is None or isinstance(obj, (int, float, str, bool, type)):
            return obj

        # Check if already visited to avoid infinite recursion
        obj_id = id(obj)
        if obj_id in visited:
            return obj
        visited.add(obj_id)

        # Handle dict
        if isinstance(obj, dict):
            return {k: _move(v) for k, v in obj.items()}

        # Handle list/tuple
        if isinstance(obj, (list, tuple)):
            return type(obj)(_move(x) for x in obj)

        # Handle objects with __dict__ (but not built-in types)
        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            try:
                for key, value in list(obj.__dict__.items()):
                    # Skip dunder attributes
                    if not key.startswith("__"):
                        setattr(obj, key, _move(value))
            except (TypeError, AttributeError):
                # Skip objects that don't allow attribute modification
                pass
            return obj

        return obj

    return cast(ModelInputT, _move(inputs))


def determine_ndim(
    first: int,
    *args: Optional[Union[int, Callable[[int], int]]],
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


PoolingMethod = Literal["mean", "max", "min", "sum", "first", "last", "hier"]


def masked_pool(
    inputs: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    pooling: Union[PoolingMethod, Sequence[PoolingMethod]] = "mean",
    normalize: bool = False,
    window_size: Optional[int] = None,
) -> torch.Tensor:
    """Apply masked pooling over the sequence dimension.

    Args:
        inputs: Input tensor of shape `(batch_size, seq_len, feature_dim)`.
        mask: Mask tensor of shape `(batch_size, seq_len)`. `True`/`1` indicates valid positions.
        pooling: Pooling method or sequence of methods.
        normalize: Whether to L2-normalize before pooling.
        window_size: Window size for hierarchical pooling (required if `pooling="hier"`).

    Returns:
        Pooled tensor of shape `(batch_size, feature_dim * num_pooling_methods)`.

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
        elif method == "hier":
            # Hierarchical pooling with sliding window
            if window_size is None:
                raise ValueError("window_size must be specified for hierarchical pooling")

            batch_size = inputs.size(0)
            feature_dim = inputs.size(-1)
            pooled_list = []

            for i in range(batch_size):
                # Get valid vectors for this sequence
                valid_vectors = inputs[i][mask[i]]
                seq_len = valid_vectors.size(0)

                if seq_len < window_size:
                    # If sequence is shorter than window, just take mean
                    pooled_list.append(valid_vectors.mean(dim=0))
                else:
                    # Slide window and compute max of means
                    output = torch.full((feature_dim,), float("-inf"), device=inputs.device)
                    for offset in range(seq_len - window_size + 1):
                        window = valid_vectors[offset : offset + window_size]
                        window_mean = window.mean(dim=0)
                        output = torch.maximum(output, window_mean)
                    pooled_list.append(output)

            pooled = torch.stack(pooled_list)
        else:
            raise ValueError(f"Unknown pooling method: {method}")

        results.append(pooled)

    return torch.cat(results, dim=-1) if len(results) > 1 else results[0]
