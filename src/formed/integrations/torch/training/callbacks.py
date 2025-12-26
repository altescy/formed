"""Training callbacks for monitoring and controlling PyTorch model training.

This module provides a callback system for PyTorch training, allowing custom
logic to be executed at various points in the training loop. Callbacks can
monitor metrics, save checkpoints, implement early stopping, and integrate
with experiment tracking systems.

Key Components:
    - TorchTrainingCallback: Base class for all callbacks
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

Examples:
    >>> from formed.integrations.torch import (
    ...     TorchTrainer,
    ...     EarlyStoppingCallback,
    ...     EvaluationCallback,
    ...     MlflowCallback
    ... )
    >>>
    >>> trainer = TorchTrainer(
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
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Optional

from colt import Registrable

from formed.workflow import use_step_logger, use_step_workdir

from ..model import BaseTorchModel
from ..types import IEvaluator, ItemT, ModelInputT, ModelOutputT, ModelParamsT
from .exceptions import StopEarly
from .state import TrainState

if TYPE_CHECKING:
    from .trainer import TorchTrainer


class TorchTrainingCallback(Registrable):
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

    Examples:
        >>> @TorchTrainingCallback.register("my_callback")
        ... class MyCallback(TorchTrainingCallback):
        ...     def on_epoch_end(self, trainer, model, state, epoch):
        ...         print(f"Completed epoch {epoch} at step {state.step}")

    """

    def on_training_start(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> None:
        pass

    def on_training_end(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> TrainState:
        return state

    def on_epoch_start(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_epoch_end(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_batch_start(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_batch_end(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        epoch: int,
        output: ModelOutputT,
    ) -> None:
        pass

    def on_eval_start(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
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
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        metrics: Mapping[str, float],
        prefix: str = "",
    ) -> None:
        pass

    def on_log(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        metrics: Mapping[str, float],
        prefix: str = "",
    ) -> None:
        pass


@TorchTrainingCallback.register("evaluation")
class EvaluationCallback(TorchTrainingCallback, Generic[ModelInputT, ModelOutputT]):
    """Callback for computing metrics using a custom evaluator.

    This callback integrates a custom evaluator into the training loop,
    resetting it before each evaluation phase and returning it for
    metric accumulation.

    Args:
        evaluator: Evaluator implementing the IEvaluator protocol.

    Examples:
        >>> from formed.integrations.ml.metrics import MulticlassAccuracy
        >>>
        >>> evaluator = MulticlassAccuracy()
        >>> callback = EvaluationCallback(evaluator)

    """

    def __init__(self, evaluator: IEvaluator[ModelInputT, ModelOutputT]) -> None:
        self._evaluator = evaluator

    def on_eval_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> IEvaluator[ModelInputT, ModelOutputT]:
        self._evaluator.reset()
        return self._evaluator


@TorchTrainingCallback.register("early_stopping")
class EarlyStoppingCallback(TorchTrainingCallback):
    """Callback for early stopping based on metric improvements.

    This callback monitors a specified metric and stops training if it
    doesn't improve for a given number of evaluations (patience). The
    best model is automatically saved and restored at the end of training.

    Args:
        patience: Number of evaluations without improvement before stopping.
        metric: Metric to monitor. Prefix with "-" to maximize (e.g., "-loss"),
            or "+" to minimize (e.g., "+error"). Default is "-loss".

    Examples:
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
        metric: str = "-train/loss",
    ) -> None:
        self._patience = patience
        self._metric = metric.lstrip("-+")
        self._direction = -1 if metric.startswith("-") else 1
        self._best_metric = -float("inf")
        self._counter = 0

    def _get_checkpoint_path(self) -> Path:
        try:
            workdir = use_step_workdir()
        except RuntimeError:
            # No workflow context, use current directory
            workdir = Path(".")
        return workdir / "best_state.pth"

    def on_training_start(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> None:
        self._best_metric = -float("inf")
        self._counter = 0

    def on_eval_end(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        metrics: Mapping[str, float],
        prefix: str = "",
    ) -> None:
        import torch
        import torch.distributed as dist

        logger = use_step_logger(__name__)

        # For DDP: only rank 0 makes the early stopping decision
        should_stop = False

        if trainer.distributor.is_main_process:
            if prefix:
                metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
            try:
                metric = self._direction * metrics[self._metric]
            except KeyError:
                return
            if metric > self._best_metric:
                self._best_metric = metric
                self._counter = 0
                # Save state_dict for serialization efficiency
                torch.save(state.state_dict(), self._get_checkpoint_path())
                logger.info(f"New best model saved with {self._metric}={self._best_metric:.4f}")
            else:
                self._counter += 1
                if self._counter >= self._patience:
                    should_stop = True

        # For DDP: broadcast the early stopping decision from rank 0 to all processes
        if trainer.distributor.world_size > 1 and dist.is_initialized():
            # Create a tensor for broadcasting
            # Get device from model's first parameter
            device = next(state.model.parameters()).device if list(state.model.parameters()) else torch.device("cpu")
            should_stop_tensor = torch.tensor(1 if should_stop else 0, dtype=torch.int, device=device)
            dist.broadcast(should_stop_tensor, src=0)
            should_stop = bool(should_stop_tensor.item())

        # Synchronize all processes before potentially raising StopEarly
        trainer.distributor.barrier()

        if should_stop:
            raise StopEarly()

    def on_training_end(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> TrainState:
        import torch
        import torch.distributed as dist

        logger = use_step_logger(__name__)

        # Synchronize before loading best model
        trainer.distributor.barrier()

        # Load best model if it exists
        if (checkpoint_path := self._get_checkpoint_path()).exists():
            if trainer.distributor.is_main_process:
                logger.info("Loading best state from early stopping checkpoint.")
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                state.load_state_dict(state_dict)

            # For DDP: broadcast the model state from rank 0 to all other processes
            if trainer.distributor.world_size > 1 and dist.is_initialized():
                # Broadcast model parameters
                for param in state.model.parameters():
                    dist.broadcast(param.data, src=0)
                # Broadcast optimizer state
                # Note: optimizer state broadcasting is complex, so we skip it for now
                # Users can re-initialize optimizer if needed after loading best model
                if trainer.distributor.is_main_process:
                    logger.info("Broadcasted best model to all processes.")

        # Synchronize after loading so all processes have the best model
        trainer.distributor.barrier()
        return state


@TorchTrainingCallback.register("mlflow")
class MlflowCallback(TorchTrainingCallback):
    """Callback for logging metrics to MLflow.

    This callback automatically logs training and validation metrics to
    MLflow when used within a workflow step that has MLflow tracking enabled.

    Examples:
        >>> from formed.integrations.torch import TorchTrainer, MlflowCallback
        >>>
        >>> trainer = TorchTrainer(
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

        self._mlflow_logger: Optional[MlflowLogger] = None

    def on_training_start(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
    ) -> None:
        from formed.integrations.mlflow.workflow import use_mlflow_logger
        from formed.workflow import use_step_logger

        # Only main process logs to MLflow
        if not trainer.distributor.is_main_process:
            return

        logger = use_step_logger(__name__)

        self._mlflow_logger = use_mlflow_logger()
        if self._mlflow_logger is None:
            logger.warning("MlflowLogger not found. Skipping logging.")

    def on_log(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        metrics: Mapping[str, float],
        prefix: str = "",
    ) -> None:
        # Only main process logs to MLflow
        if not trainer.distributor.is_main_process:
            return

        metrics = {prefix + key: value for key, value in metrics.items()}
        if self._mlflow_logger is not None:
            # Log all metrics
            for key, value in metrics.items():
                self._mlflow_logger.log_metric(key, value, step=int(state.step))

            # Log learning rate
            learning_rate = state.get_learning_rate()
            if learning_rate is not None:
                self._mlflow_logger.log_metric("learning_rate", learning_rate, step=int(state.step))

            # Log gradient norm
            gradient_norm = state.get_gradient_norm()
            if gradient_norm is not None:
                self._mlflow_logger.log_metric("gradient_norm", gradient_norm, step=int(state.step))

    def on_epoch_end(
        self,
        trainer: "TorchTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        epoch: int,
    ) -> None:
        # Only main process logs to MLflow
        if not trainer.distributor.is_main_process:
            return

        if self._mlflow_logger is not None:
            self._mlflow_logger.log_metric("epoch", epoch, step=int(state.step))
