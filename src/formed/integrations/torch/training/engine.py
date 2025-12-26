"""Training engine abstractions for PyTorch models.

This module provides the training engine abstraction that defines how
models are trained and evaluated. Engines handle loss computation,
gradient calculation, and parameter updates.

Key Components:
    - TorchTrainingEngine: Abstract base class for training engines
    - DefaultTorchTrainingEngine: Default implementation with automatic differentiation

Features:
    - Customizable loss functions
    - Automatic gradient computation using PyTorch autograd
    - State creation and management
    - Separate train and eval steps
    - Compatible with TorchTrainer and distributors

Examples:
    >>> from formed.integrations.torch import DefaultTorchTrainingEngine
    >>>
    >>> # Create engine with custom loss accessor
    >>> engine = DefaultTorchTrainingEngine(loss="total_loss")
    >>>
    >>> # Or with custom loss function
    >>> def custom_loss(output):
    ...     return output.loss + 0.1 * output.regularization
    >>> engine = DefaultTorchTrainingEngine(loss=custom_loss)

"""

import abc
import warnings
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, Literal, Optional, Union, cast

import torch
from colt import Lazy, Registrable

from formed.common.attributeutils import xgetattr

from ..context import get_device
from ..model import BaseTorchModel
from ..types import (
    IGradScaler,
    ILRScheduler,
    IOptimizer,
    LRSchedulerFactory,
    ModelInputT,
    ModelOutputT,
    ModelParamsT,
    OptimizerFactory,
)
from .state import TrainState

if TYPE_CHECKING:
    from .trainer import TorchTrainer


class TorchTrainingEngine(abc.ABC, Registrable, Generic[ModelInputT, ModelOutputT, ModelParamsT]):
    """Abstract base class for PyTorch training engines.

    A training engine defines how models are trained by implementing
    state creation, training steps, and evaluation steps. This allows
    for custom training loops and loss computations.

    Type Parameters:
        ModelInputT: Type of model input.
        ModelOutputT: Type of model output.
        ModelParamsT: Type of additional parameters.

    """

    @abc.abstractmethod
    def create_state(
        self,
        trainer: "TorchTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
    ) -> TrainState:
        """Create initial training state from model and trainer.

        Args:
            trainer: Trainer instance.
            model: Model to train.

        Returns:
            Initial training state.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "TorchTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
    ) -> ModelOutputT:
        """Execute a single training step.

        Args:
            inputs: Batch of training inputs.
            state: Current training state (model and optimizer are updated in-place).
            trainer: Trainer instance.

        Returns:
            Model output.

        Note:
            This method updates the state in-place for efficiency.
            The step counter is incremented automatically.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def eval_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "TorchTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
    ) -> ModelOutputT:
        """Execute a single evaluation step.

        Args:
            inputs: Batch of evaluation inputs.
            state: Current training state.
            trainer: Trainer instance.

        Returns:
            Model output.

        """
        raise NotImplementedError


def get_default_optimizer_factory() -> OptimizerFactory:
    """Get a default optimizer factory (Adam with lr=1e-3)."""
    return Lazy(config={"lr": 1e-3}, cls=torch.optim.Adam)


def get_default_lr_scheduler_factory() -> Optional[LRSchedulerFactory]:
    """Get a default learning rate scheduler factory (None)."""
    return None


@TorchTrainingEngine.register("default")
class DefaultTorchTrainingEngine(TorchTrainingEngine[ModelInputT, ModelOutputT, ModelParamsT]):
    """Default training engine using automatic differentiation.

    This engine computes gradients using PyTorch's autograd
    and updates parameters using the provided optimizer. Loss is extracted
    from model output either by attribute name or custom function.

    Args:
        optimizer: Optimizer factory or instance. Can be a Lazy object, callable
            that takes model parameters, or an optimizer instance.
        lr_scheduler: Optional learning rate scheduler factory or instance.
            Can be a Lazy object, callable that takes optimizer, a sequence of
            schedulers (will be chained), or a scheduler instance.
        loss: Loss accessor - either attribute name (e.g., "loss") or
            callable that extracts loss from model output.
        gradient_accumulation_steps: Number of steps to accumulate gradients
            before performing an optimizer step.
        max_grad_norm: Maximum gradient norm for clipping. If None, no clipping is applied.
        params: Optional additional parameters to pass to the model during training.
        dtype: Data type for mixed precision training ("float32", "float16", "bfloat16").
        grad_scaler: Gradient scaler for mixed precision training.

    Examples:
        >>> # Basic usage with optimizer
        >>> engine = DefaultTorchTrainingEngine(
        ...     optimizer=torch.optim.Adam,
        ...     loss="loss"
        ... )
        >>>
        >>> # With learning rate scheduler and gradient clipping
        >>> engine = DefaultTorchTrainingEngine(
        ...     optimizer=Lazy(cls=torch.optim.Adam, config={"lr": 1e-3}),
        ...     lr_scheduler=Lazy(cls=torch.optim.lr_scheduler.CosineAnnealingLR, config={"T_max": 100}),
        ...     max_grad_norm=1.0,
        ...     loss=lambda output: output.loss + 0.01 * output.regularization
        ... )
        >>>
        >>> # With mixed precision training
        >>> engine = DefaultTorchTrainingEngine(
        ...     optimizer=torch.optim.AdamW,
        ...     dtype="bfloat16",
        ...     grad_scaler=Lazy(cls=torch.amp.GradScaler),
        ...     max_grad_norm=1.0
        ... )

    """

    def __init__(
        self,
        optimizer: Optional[OptimizerFactory] = None,
        lr_scheduler: Union[LRSchedulerFactory, Sequence[LRSchedulerFactory], None] = None,
        loss: Union[str, Callable[[ModelOutputT], torch.Tensor]] = "loss",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        params: ModelParamsT | None = None,
        dtype: Literal["float32", "float16", "bfloat16"] | None = None,
        grad_scaler: Lazy[torch.amp.grad_scaler.GradScaler | IGradScaler] | IGradScaler | None = None,
    ) -> None:
        assert dtype in (None, "float32", "float16", "bfloat16"), (
            "dtype must be one of None, 'float32', 'float16', or 'bfloat16'"
        )

        super().__init__()
        self._optimizer_factory = optimizer or get_default_optimizer_factory()
        self._lr_scheduler_factory = lr_scheduler or get_default_lr_scheduler_factory()
        self._loss = partial(xgetattr, name=loss) if isinstance(loss, str) else loss
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._max_grad_norm = max_grad_norm
        self._params = params
        self._dtype = getattr(torch, dtype) if dtype is not None else None
        self._grad_scaler = grad_scaler

    def create_state(
        self,
        trainer: "TorchTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
    ) -> TrainState:
        # Construct optimizer
        optimizer: IOptimizer
        if isinstance(self._optimizer_factory, Lazy):
            optimizer = self._optimizer_factory.construct(params=model.parameters())
        elif callable(self._optimizer_factory):
            optimizer = self._optimizer_factory(model.parameters())
        else:
            optimizer = self._optimizer_factory

        # Construct lr_scheduler
        lr_scheduler: ILRScheduler | None = None
        if self._lr_scheduler_factory is not None:

            def _construct_lr_scheduler(factory: LRSchedulerFactory) -> ILRScheduler:
                if isinstance(factory, Lazy):
                    return factory.construct(optimizer=optimizer)
                elif callable(factory):
                    return factory(optimizer)
                return factory

            if isinstance(self._lr_scheduler_factory, Sequence):
                lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        cast(torch.optim.lr_scheduler.LRScheduler, _construct_lr_scheduler(scheduler))
                        for scheduler in self._lr_scheduler_factory
                    ],
                    optimizer=cast(torch.optim.Optimizer, optimizer),
                )
            else:
                lr_scheduler = _construct_lr_scheduler(self._lr_scheduler_factory)

        # Initialize grad_scaler only if requested and on appropriate device
        grad_scaler: IGradScaler | None = None
        if isinstance(self._grad_scaler, IGradScaler):
            grad_scaler = self._grad_scaler
        elif isinstance(self._grad_scaler, Lazy):
            if device := get_device():
                grad_scaler = self._grad_scaler.construct(device=device.type)
            else:
                warnings.warn("GradScaler requested but device is not set. GradScaler will not be used.")

        return TrainState(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            step=0,
            grad_scaler=grad_scaler,
        )

    def train_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "TorchTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
    ) -> ModelOutputT:
        del trainer

        # Set model to training mode
        state.model.train()

        # Zero gradients at the start of accumulation cycle
        if state.step % self._gradient_accumulation_steps == 0:
            state.optimizer.zero_grad()

        with ExitStack() as stack:
            if (device := get_device()) is not None and self._dtype is not None:
                stack.enter_context(torch.autocast(device_type=device.type, dtype=self._dtype))

            output = state.model(inputs, params=self._params)

            try:
                loss = self._loss(output)
            except (KeyError, AttributeError) as e:
                raise ValueError(
                    f"Failed to extract loss from model output. "
                    f"Error: {e}. "
                    f"Output type: {type(output).__name__}. "
                    "Please ensure your model's forward() method returns output with a 'loss' attribute or key."
                ) from e

        if loss is None:
            raise ValueError(
                "Model output loss is None. "
                "This typically happens when labels are not provided during training. "
                "Please ensure your training data includes labels."
            )

        # Scale loss for gradient accumulation
        loss = loss / self._gradient_accumulation_steps

        # Backward pass with or without gradient scaling
        if state.grad_scaler is not None:
            state.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update optimizer and scheduler when accumulation is complete
        if (state.step + 1) % self._gradient_accumulation_steps == 0:
            # Clip gradients if max_grad_norm is specified
            if self._max_grad_norm is not None:
                if state.grad_scaler is not None:
                    if isinstance(state.optimizer, torch.optim.Optimizer):
                        # Unscale gradients before clipping when using grad_scaler
                        state.grad_scaler.unscale_(state.optimizer)
                    else:
                        warnings.warn(
                            "Cannot unscale gradients for gradient clipping because "
                            "the optimizer is not a torch.optim.Optimizer instance."
                        )
                torch.nn.utils.clip_grad_norm_(state.model.parameters(), self._max_grad_norm)

            if state.grad_scaler is not None:
                # GradScaler.step expects torch.optim.Optimizer
                state.grad_scaler.step(cast(torch.optim.Optimizer, state.optimizer))
                state.grad_scaler.update()
            else:
                state.optimizer.step()
            if state.lr_scheduler is not None:
                state.lr_scheduler.step()

        # Increment step counter (counts micro-batches)
        state.step += 1

        return output

    def eval_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "TorchTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
    ) -> ModelOutputT:
        del trainer

        # Set model to eval mode
        state.model.eval()

        # Standard PyTorch evaluation step
        with torch.no_grad():
            output = state.model(inputs)

        return output
