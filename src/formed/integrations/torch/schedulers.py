"""Learning rate schedulers for PyTorch models.

This module provides custom learning rate schedulers that extend PyTorch's
standard scheduler functionality, including cosine annealing with warm restarts
and warmup phases.

Available Schedulers:
    - CosineLRScheduler: Cosine annealing with optional restarts and warmup

Features:
    - Cosine decay with configurable cycle length
    - Warm restarts with cycle multiplier
    - Learning rate warmup phase
    - Cycle-based decay multiplier
    - Compatible with Colt registration system

Examples:
    >>> from formed.integrations.torch.schedulers import CosineLRScheduler
    >>>
    >>> scheduler = CosineLRScheduler(
    ...     optimizer,
    ...     t_initial=100,
    ...     lr_min=1e-6,
    ...     warmup_t=5,
    ...     warmup_lr_init=1e-5
    ... )
    >>> for epoch in range(num_epochs):
    ...     train(...)
    ...     scheduler.step(epoch + 1)

"""

import math
from typing import Any

import torch.optim as optim


class CosineLRScheduler(optim.lr_scheduler.LRScheduler):
    """Cosine annealing learning rate scheduler with warm restarts.

    Implements the SGDR (Stochastic Gradient Descent with Warm Restarts)
    algorithm described in https://arxiv.org/abs/1608.03983.

    This scheduler decreases the learning rate following a cosine curve,
    optionally restarting the schedule multiple times during training.
    It also supports a warmup phase at the beginning.

    Args:
        optimizer: Wrapped optimizer.
        t_initial: Number of iterations/epochs for the first cycle.
        lr_min: Minimum learning rate. Default: 0.
        cycle_mul: Multiplier for cycle length after each restart. Default: 1.0.
        cycle_decay: Decay factor applied to learning rate at each restart. Default: 1.0.
        cycle_limit: Maximum number of restart cycles (0 means no limit). Default: 1.
        warmup_t: Number of warmup iterations/epochs. Default: 0.
        warmup_lr_init: Initial learning rate during warmup. Default: 0.
        warmup_prefix: If True, warmup iterations don't count toward t_initial. Default: False.
        t_in_epochs: If True, t values are in epochs; otherwise in iterations. Default: True.
        last_epoch: The index of last epoch. Default: -1.

    Examples:
        >>> # Create scheduler with 100 epoch cycles and 5 epoch warmup
        >>> scheduler = CosineLRScheduler(
        ...     optimizer,
        ...     t_initial=100,
        ...     lr_min=1e-6,
        ...     cycle_mul=2.0,  # Each cycle is 2x longer
        ...     warmup_t=5,
        ...     warmup_lr_init=1e-5
        ... )
        >>>
        >>> # Update learning rate each epoch
        >>> for epoch in range(num_epochs):
        ...     train_one_epoch(...)
        ...     scheduler.step(epoch + 1)

    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        t_initial: int,
        lr_min: float = 0.0,
        cycle_mul: float = 1.0,
        cycle_decay: float = 1.0,
        cycle_limit: int = 1,
        warmup_t: int = 0,
        warmup_lr_init: float = 0.0,
        warmup_prefix: bool = False,
        t_in_epochs: bool = True,
        last_epoch: int = -1,
    ) -> None:
        assert t_initial > 0, "t_initial must be positive"
        assert lr_min >= 0, "lr_min must be non-negative"

        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs

        # Store base learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        # Initialize warmup steps
        if self.warmup_t:
            self.warmup_steps = [(base_lr - warmup_lr_init) / self.warmup_t for base_lr in self.base_lrs]
        else:
            self.warmup_steps = [1.0 for _ in self.base_lrs]

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rate at the current step.

        Returns:
            List of learning rates for each parameter group.

        """
        # Current timestep (starts from 0 after first step())
        t = self.last_epoch

        # Warmup phase: linearly interpolate from warmup_lr_init to base_lr
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + (t + 1) * step for step in self.warmup_steps]
            return lrs

        # Adjust t if warmup is a prefix
        if self.warmup_prefix:
            t = t - self.warmup_t

        # Determine current cycle
        if self.cycle_mul == 1.0:
            # Simple case: equal cycles
            cycle = t // self.t_initial
            t_curr = t % self.t_initial
            t_i = self.t_initial
        else:
            # Geometric progression of cycle lengths
            # Find which cycle we're in using logarithmic calculation
            cycle = int(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
            # Compute cumulative time up to current cycle
            t_prev = self.t_initial * (1 - self.cycle_mul**cycle) / (1 - self.cycle_mul)
            t_curr = t - t_prev
            t_i = self.t_initial * (self.cycle_mul**cycle)

        # Apply cycle limit
        if self.cycle_limit > 0 and cycle >= self.cycle_limit:
            return [self.lr_min for _ in self.base_lrs]

        # Compute cycle decay
        cycle_decay = self.cycle_decay**cycle

        # Cosine annealing
        lrs = [
            self.lr_min + (base_lr - self.lr_min) * cycle_decay * 0.5 * (1 + math.cos(math.pi * t_curr / t_i))
            for base_lr in self.base_lrs
        ]

        return lrs

    def get_cycle_length(self, cycles: int = 0) -> int:
        """Calculate total number of iterations for a given number of cycles.

        Args:
            cycles: Number of cycles (0 means current cycle).

        Returns:
            Total number of iterations.

        """
        if cycles <= 0:
            cycles = self.cycle_limit if self.cycle_limit > 0 else 1

        if self.cycle_mul == 1.0:
            length = self.t_initial * cycles
        else:
            length = int(self.t_initial * (1 - self.cycle_mul**cycles) / (1 - self.cycle_mul))

        if self.warmup_prefix:
            length += self.warmup_t

        return length

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a dict.

        Returns:
            Dictionary containing scheduler state.

        """
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_get_lr_called_within_step", "_step_count")
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler state.

        Args:
            state_dict: Scheduler state dict.

        """
        self.__dict__.update(state_dict)
