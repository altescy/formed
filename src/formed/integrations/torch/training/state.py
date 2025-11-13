"""Training state management for PyTorch models.

This module provides a training state class that encapsulates model parameters,
optimizer state, and training progress for PyTorch models.

"""

from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from ..types import IOptimizer


class TrainState:
    """Training state for PyTorch models.

    This class encapsulates the training state including model,
    optimizer, and training progress counters. Unlike the Flax version,
    this directly holds references to the model and optimizer for efficiency.

    Attributes:
        model: The PyTorch model being trained.
        optimizer: The optimizer for training.
        step: Training step counter.

    Example:
        >>> # Create state from model and optimizer
        >>> state = TrainState(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     step=0
        ... )
        >>>
        >>> # Access model and optimizer directly
        >>> state.model.train()
        >>> state.optimizer.zero_grad()

    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: "IOptimizer",
        step: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.step = step

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization.

        Returns:
            Dictionary containing model state, optimizer state, and step.

        """
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": self.step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from dictionary.

        Args:
            state_dict: Dictionary containing model state, optimizer state, and step.

        """
        self.model.load_state_dict(state_dict["model_state"])
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        self.step = state_dict["step"]
