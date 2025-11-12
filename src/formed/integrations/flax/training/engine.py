"""Training engine abstractions for Flax models.

This module provides the training engine abstraction that defines how
models are trained and evaluated. Engines handle loss computation,
gradient calculation, and parameter updates.

Key Components:
    - FlaxTrainingEngine: Abstract base class for training engines
    - DefaultFlaxTrainingEngine: Default implementation with automatic differentiation

Features:
    - Customizable loss functions
    - Automatic gradient computation using JAX
    - State creation and management
    - Separate train and eval steps
    - Compatible with FlaxTrainer and distributors

Example:
    >>> from formed.integrations.flax import DefaultFlaxTrainingEngine
    >>>
    >>> # Create engine with custom loss accessor
    >>> engine = DefaultFlaxTrainingEngine(loss="total_loss")
    >>>
    >>> # Or with custom loss function
    >>> def custom_loss(output):
    ...     return output.loss + 0.1 * output.regularization
    >>> engine = DefaultFlaxTrainingEngine(loss=custom_loss)

"""

import abc
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, cast

import jax
from colt import Registrable
from flax import nnx

from formed.common.attributeutils import xgetattr

from ..model import BaseFlaxModel
from ..types import ModelInputT, ModelOutputT, ModelParamsT
from .state import TrainState

if TYPE_CHECKING:
    from .trainer import FlaxTrainer


class FlaxTrainingEngine(abc.ABC, Registrable, Generic[ModelInputT, ModelOutputT, ModelParamsT]):
    """Abstract base class for Flax training engines.

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
        rngs: nnx.Rngs,
        trainer: "FlaxTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
    ) -> TrainState:
        """Create initial training state from model and trainer.

        Args:
            rngs: Random number generators.
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
        trainer: "FlaxTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
    ) -> tuple[TrainState, ModelOutputT]:
        """Execute a single training step.

        Args:
            inputs: Batch of training inputs.
            state: Current training state.
            trainer: Trainer instance.

        Returns:
            Tuple of (updated_state, model_output).

        """
        raise NotImplementedError

    @abc.abstractmethod
    def eval_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "FlaxTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
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


@FlaxTrainingEngine.register("default")
class DefaultFlaxTrainingEngine(FlaxTrainingEngine[ModelInputT, ModelOutputT, ModelParamsT]):
    """Default training engine using automatic differentiation.

    This engine computes gradients using JAX's automatic differentiation
    and updates parameters using the provided optimizer. Loss is extracted
    from model output either by attribute name or custom function.

    Args:
        loss: Loss accessor - either attribute name (e.g., "loss") or
            callable that extracts loss from model output.

    Example:
        >>> # Use output.loss attribute
        >>> engine = DefaultFlaxTrainingEngine(loss="loss")
        >>>
        >>> # Use custom loss function
        >>> engine = DefaultFlaxTrainingEngine(
        ...     loss=lambda output: output.loss + 0.01 * output.regularization
        ... )

    """

    def __init__(self, loss: str | Callable[[ModelOutputT], jax.Array] = "loss") -> None:
        super().__init__()
        self._loss = partial(xgetattr, name=loss) if isinstance(loss, str) else loss

    def create_state(
        self,
        rngs: nnx.Rngs,
        trainer: "FlaxTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
    ) -> TrainState:
        graphdef, params, *states = nnx.split(model, nnx.Param, nnx.BatchStat, nnx.RngState)
        return cast(
            TrainState,
            TrainState.create(
                apply_fn=None,
                graphdef=graphdef,
                additional_states=tuple(states),
                params=params,
                tx=trainer.optimizer,
            ),
        )

    def train_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "FlaxTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
    ) -> tuple[TrainState, ModelOutputT]:
        def step(state: TrainState, inputs: ModelInputT) -> tuple[TrainState, ModelOutputT]:
            model = nnx.merge(state.graphdef, state.params, *state.additional_states)
            model.train()

            def loss_fn(
                model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
            ) -> tuple[jax.Array, ModelOutputT]:
                output = model(inputs)
                loss = self._loss(output)
                return loss, output

            (_, output), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

            graphdef, params, *additional_states = nnx.split(model, nnx.Param, nnx.BatchStat, nnx.RngState)

            grads = trainer.distributor.reduce(grads)
            state = state.replace(
                graphdef=graphdef,
                params=params,
                additional_states=tuple(additional_states),
            )
            state = state.apply_gradients(grads=grads)
            return state, output

        return step(state, inputs)

    def eval_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "FlaxTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
    ) -> ModelOutputT:
        del trainer

        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT] = nnx.merge(
            state.graphdef,
            state.params,
            *state.additional_states,
        )
        model.eval()
        return model(inputs)
