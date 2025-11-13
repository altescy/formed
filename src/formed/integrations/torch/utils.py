"""Utility functions for PyTorch integration."""

import torch

from .types import TensorCompatible


def ensure_torch_tensor(x: TensorCompatible, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert array-like objects to PyTorch tensors.

    Args:
        x: Input data (tensor, numpy array, list, etc.)
        dtype: Optional dtype for the output tensor.

    Returns:
        PyTorch tensor.

    Example:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> tensor = ensure_torch_tensor(arr)
        >>> type(tensor)
        <class 'torch.Tensor'>
    """
    if isinstance(x, torch.Tensor):
        if dtype is not None and x.dtype != dtype:
            return x.to(dtype)
        return x

    # Convert numpy arrays, handling float64 -> float32 conversion
    import numpy as np

    if isinstance(x, np.ndarray):
        if dtype is None and x.dtype == np.float64:
            # Default: convert float64 to float32 for PyTorch
            dtype = torch.float32
        return torch.from_numpy(x).to(dtype) if dtype is not None else torch.from_numpy(x)

    # Convert other array-like objects
    tensor = torch.as_tensor(x)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor
