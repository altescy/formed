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

Example:
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
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Generic

import torch
from colt import Registrable

from formed.common.attributeutils import xgetattr

from ..model import BaseTorchModel
from ..types import ModelInputT, ModelOutputT, ModelParamsT
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


@TorchTrainingEngine.register("default")
class DefaultTorchTrainingEngine(TorchTrainingEngine[ModelInputT, ModelOutputT, ModelParamsT]):
    """Default training engine using automatic differentiation.

    This engine computes gradients using PyTorch's autograd
    and updates parameters using the provided optimizer. Loss is extracted
    from model output either by attribute name or custom function.

    Args:
        loss: Loss accessor - either attribute name (e.g., "loss") or
            callable that extracts loss from model output.

    Example:
        >>> # Use output.loss attribute
        >>> engine = DefaultTorchTrainingEngine(loss="loss")
        >>>
        >>> # Use custom loss function
        >>> engine = DefaultTorchTrainingEngine(
        ...     loss=lambda output: output.loss + 0.01 * output.regularization
        ... )

    """

    def __init__(self, loss: str | Callable[[ModelOutputT], torch.Tensor] = "loss") -> None:
        super().__init__()
        self._loss = partial(xgetattr, name=loss) if isinstance(loss, str) else loss

    def create_state(
        self,
        trainer: "TorchTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
    ) -> TrainState:
        return TrainState(
            model=model,
            optimizer=trainer.optimizer,
            step=0,
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

        # Standard PyTorch training step
        state.optimizer.zero_grad()
        output = state.model(inputs)

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

        loss.backward()
        state.optimizer.step()

        # Increment step counter
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
