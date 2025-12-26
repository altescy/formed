"""Training state management for PyTorch models.

This module provides a training state class that encapsulates model parameters,
optimizer state, and training progress for PyTorch models.

"""

from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from ..types import IGradScaler, ILRScheduler, IOptimizer


class TrainState:
    """Training state for PyTorch models.

    This class encapsulates the training state including model,
    optimizer, learning rate scheduler, and training progress counters.
    Unlike the Flax version, this directly holds references to the model
    and optimizer for efficiency.

    Attributes:
        model: The PyTorch model being trained.
        optimizer: The optimizer for training.
        lr_scheduler: Optional learning rate scheduler.
        step: Training step counter.
        grad_scaler: Optional gradient scaler for mixed precision training.

    Examples:
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
        model: torch.nn.Module,
        optimizer: "IOptimizer",
        step: int = 0,
        lr_scheduler: Optional["ILRScheduler"] = None,
        grad_scaler: Optional["IGradScaler"] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.grad_scaler = grad_scaler

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization.

        Returns:
            Dictionary containing model state, `optimizer` state, `lr_scheduler` state (if present), `grad_scaler` state (if present), and `step`.

        """
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": self.step,
        }
        if self.lr_scheduler is not None:
            state["lr_scheduler_state"] = self.lr_scheduler.state_dict()
        if self.grad_scaler is not None:
            state["grad_scaler_state"] = self.grad_scaler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from dictionary.

        Args:
            state_dict: Dictionary containing model state, optimizer state, lr_scheduler state (optional), grad_scaler state (optional), and step.

        """
        self.model.load_state_dict(state_dict["model_state"])
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        self.step = state_dict["step"]
        if "lr_scheduler_state" in state_dict and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler_state"])
        if "grad_scaler_state" in state_dict and self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler_state"])

    def get_learning_rate(self) -> Optional[float]:
        """Get current learning rate from optimizer.

        Returns:
            Current learning rate from the first parameter group, or `None` if unavailable.

        Examples:
            >>> lr = state.get_learning_rate()
            >>> if lr is not None:
            ...     print(f"Current learning rate: {lr}")

        """
        import torch.optim

        if isinstance(self.optimizer, torch.optim.Optimizer):
            if self.optimizer.param_groups:
                return self.optimizer.param_groups[0]["lr"]
        return None

    def get_gradient_norm(self) -> Optional[float]:
        """Compute L2 norm of all gradients.

        Returns:
            L2 norm of all parameter gradients, or `None` if no gradients are available.

        Examples:
            >>> grad_norm = state.get_gradient_norm()
            >>> if grad_norm is not None:
            ...     print(f"Gradient norm: {grad_norm:.4f}")

        Note:
            This method computes the gradient norm on-demand. It should be called
            after `backward()` but before optimizer.step() or `zero_grad()` to get
            meaningful results.

        """
        total_norm = 0.0
        has_gradients = False

        for param in self.model.parameters():
            if param.grad is not None:
                has_gradients = True
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        if not has_gradients:
            return None

        return total_norm**0.5
