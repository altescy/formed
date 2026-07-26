"""Shared state contracts for stateful PyTorch modules."""

from typing import Protocol, Self, runtime_checkable

import torch


@runtime_checkable
class ReorderableState(Protocol):
    """State that can follow selection, duplication, and reordering operations."""

    def reorder(self, indices: torch.Tensor) -> Self:
        """Select, reorder, or duplicate states using flattened batch indices."""
        ...
