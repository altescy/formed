"""Training callbacks for monitoring and controlling Flax model training.

This module provides a callback system for Flax training, allowing custom
logic to be executed at various points in the training loop. Callbacks can
monitor metrics, save checkpoints, implement early stopping, and integrate
with experiment tracking systems.

Key Components:
    - FlaxTrainingCallback: Base class for all callbacks
    - EvaluationCallback: Computes metrics using custom evaluators
    - EarlyStoppingCallback: Stops training based on metric improvements
    - MlflowCallback: Logs metrics to MLflow

Features:
    - Hook points at training/epoch/batch start and end
    - Metric computation and logging
    - Model checkpointing
    - Early stopping with patience
    - MLflow integration
    - Extensible for custom callbacks

Example:
    >>> from formed.integrations.flax import (
    ...     FlaxTrainer,
    ...     EarlyStoppingCallback,
    ...     EvaluationCallback,
    ...     MlflowCallback
    ... )
    >>>
    >>> trainer = FlaxTrainer(
    ...     train_dataloader=train_loader,
    ...     val_dataloader=val_loader,
    ...     callbacks=[
    ...         EvaluationCallback(my_evaluator),
    ...         EarlyStoppingCallback(patience=5, metric="-loss"),
    ...         MlflowCallback()
    ...     ]
    ... )

"""

from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Generic, cast

import orbax.checkpoint
from colt import Registrable

from formed.workflow import use_step_logger, use_step_workdir

from ..model import BaseFlaxModel
from ..types import IEvaluator, ItemT, ModelInputT, ModelOutputT, ModelParamsT
from .exceptions import StopEarly
from .state import TrainState

if TYPE_CHECKING:
    from .trainer import FlaxTrainer


class FlaxTrainingCallback(Registrable):
    """Base class for training callbacks.

    Callbacks provide hooks to execute custom logic at various points
    during training. Subclasses can override any hook method to implement
    custom behavior such as logging, checkpointing, or early stopping.

    Hook execution order:
        1. on_training_start - once at the beginning
        2. on_epoch_start - at the start of each epoch
        3. on_batch_start - before each training batch
        4. on_batch_end - after each training batch
        5. on_eval_start - before evaluation (returns evaluator)
        6. on_eval_end - after evaluation with computed metrics
        7. on_log - when metrics are logged
        8. on_epoch_end - at the end of each epoch
        9. on_training_end - once at the end (can modify final state)

    Example:
        >>> @FlaxTrainingCallback.register("my_callback")
        ... class MyCallback(FlaxTrainingCallback):
        ...     def on_epoch_end(self, trainer, model, state, epoch):
        ...         print(f"Completed epoch {epoch} at step {state.step}")

    """

    def on_training_start(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> None:
        pass

    def on_training_end(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> TrainState:
        return state

    def on_epoch_start(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_epoch_end(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_batch_start(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_batch_end(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        epoch: int,
        output: ModelOutputT,
    ) -> None:
        pass

    def on_eval_start(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> IEvaluator[ModelInputT, ModelOutputT]:
        class DummyMetric(IEvaluator):
            def update(self, inputs, output, /) -> None:
                pass

            def compute(self) -> dict[str, float]:
                return {}

            def reset(self) -> None:
                pass

        return DummyMetric()

    def on_eval_end(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        metrics: Mapping[str, float],
        prefix: str = "",
    ) -> None:
        pass

    def on_log(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        metrics: Mapping[str, float],
        prefix: str = "",
    ) -> None:
        pass


@FlaxTrainingCallback.register("evaluation")
class EvaluationCallback(FlaxTrainingCallback, Generic[ModelInputT, ModelOutputT]):
    """Callback for computing metrics using a custom evaluator.

    This callback integrates a custom evaluator into the training loop,
    resetting it before each evaluation phase and returning it for
    metric accumulation.

    Args:
        evaluator: Evaluator implementing the IEvaluator protocol.

    Example:
        >>> from formed.integrations.ml.metrics import MulticlassAccuracy
        >>>
        >>> evaluator = MulticlassAccuracy()
        >>> callback = EvaluationCallback(evaluator)

    """

    def __init__(self, evaluator: IEvaluator[ModelInputT, ModelOutputT]) -> None:
        self._evaluator = evaluator

    def on_eval_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> IEvaluator[ModelInputT, ModelOutputT]:
        self._evaluator.reset()
        return self._evaluator


@FlaxTrainingCallback.register("early_stopping")
class EarlyStoppingCallback(FlaxTrainingCallback):
    """Callback for early stopping based on metric improvements.

    This callback monitors a specified metric and stops training if it
    doesn't improve for a given number of evaluations (patience). The
    best model is automatically saved and restored at the end of training.

    Args:
        patience: Number of evaluations without improvement before stopping.
        metric: Metric to monitor. Prefix with "-" to maximize (e.g., "-loss"),
            or "+" to minimize (e.g., "+error"). Default is "-loss".

    Example:
        >>> # Stop if validation loss doesn't improve for 5 evaluations
        >>> callback = EarlyStoppingCallback(patience=5, metric="-val/loss")
        >>>
        >>> # Stop if accuracy doesn't improve for 3 evaluations
        >>> callback = EarlyStoppingCallback(patience=3, metric="+accuracy")

    Note:
        The best model is saved to the step working directory and
        automatically restored when training ends early or completes.

    """

    def __init__(
        self,
        patience: int = 5,
        metric: str = "-loss",
    ) -> None:
        self._patience = patience
        self._metric = metric.lstrip("-+")
        self._direction = -1 if metric.startswith("-") else 1
        self._best_metric = -float("inf")
        self._best_step: int | None = None
        self._counter = 0

    @cached_property
    def _checkpointer(self) -> orbax.checkpoint.CheckpointManager:
        workdir = use_step_workdir()
        return orbax.checkpoint.CheckpointManager(
            workdir / "best_checkpoint",
            orbax.checkpoint.PyTreeCheckpointer(),
            options=orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True),
        )

    def _get_best_checkpoint_path(self) -> Path:
        workdir = use_step_workdir()
        return workdir / "best_checkpoint"

    def on_training_start(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> None:
        self._best_metric = -float("inf")
        self._counter = 0

    def on_eval_end(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        metrics: Mapping[str, float],
        prefix: str = "",
    ) -> None:
        logger = use_step_logger(__name__)
        if prefix:
            metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
        try:
            metric = self._direction * metrics[self._metric]
        except KeyError:
            return

        if metric > self._best_metric:
            self._best_metric = metric
            self._best_step = int(state.step)
            self._counter = 0
            self._checkpointer.save(int(state.step), state)
            logger.info(f"New best model saved with {self._metric}={self._best_metric:.4f}")
        else:
            self._counter += 1
            if self._counter >= self._patience:
                raise StopEarly()

    def on_training_end(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> TrainState:
        logger = use_step_logger(__name__)
        if self._best_step is not None:
            logger.info("Restoring best state from early stopping checkpoint.")
            state = cast(TrainState, self._checkpointer.restore(self._best_step, items=state))
        return state


@FlaxTrainingCallback.register("mlflow")
class MlflowCallback(FlaxTrainingCallback):
    """Callback for logging metrics to MLflow.

    This callback automatically logs training and validation metrics to
    MLflow when used within a workflow step that has MLflow tracking enabled.

    Example:
        >>> from formed.integrations.flax import FlaxTrainer, MlflowCallback
        >>>
        >>> trainer = FlaxTrainer(
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     callbacks=[MlflowCallback()]
        ... )

    Note:
        Requires the formed mlflow integration and must be used within
        a workflow step with MLflow tracking configured.

    """

    def __init__(self) -> None:
        from formed.integrations.mlflow.workflow import MlflowLogger

        self._mlflow_logger: MlflowLogger | None = None

    def on_training_start(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> None:
        from formed.integrations.mlflow.workflow import use_mlflow_logger
        from formed.workflow import use_step_logger

        logger = use_step_logger(__name__)

        self._mlflow_logger = use_mlflow_logger()
        if self._mlflow_logger is None:
            logger.warning("MlflowLogger not found. Skipping logging.")

    def on_log(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        metrics: Mapping[str, float],
        prefix: str = "",
    ) -> None:
        metrics = {prefix + key: value for key, value in metrics.items()}
        if self._mlflow_logger is not None:
            for key, value in metrics.items():
                self._mlflow_logger.log_metric(key, value, step=int(state.step))

    def on_epoch_end(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        epoch: int,
    ) -> None:
        if self._mlflow_logger is not None:
            self._mlflow_logger.log_metric("epoch", epoch, step=int(state.step))
