"""High-level trainer for PyTorch models.

This module provides the TorchTrainer class, which orchestrates the complete
training process for PyTorch models including data loading, optimization,
evaluation, callbacks, and distributed training.

Key Features:
    - Flexible training loop with epoch and step-based logging/evaluation
    - Support for callbacks at various training stages
    - Distributed training via data parallelism
    - Integration with PyTorch optimizers
    - Rich progress bars with training metrics
    - Early stopping and checkpointing
    - MLflow integration

Example:
    >>> from formed.integrations.torch import (
    ...     TorchTrainer,
    ...     EvaluationCallback,
    ...     EarlyStoppingCallback
    ... )
    >>> from formed.integrations.ml import DataLoader, BasicBatchSampler
    >>> import torch.optim as optim
    >>>
    >>> # Setup data loaders
    >>> train_dataloader = DataLoader(
    ...     sampler=BasicBatchSampler(batch_size=32, shuffle=True),
    ...     collator=datamodule.batch
    ... )
    >>>
    >>> # Create trainer
    >>> trainer = TorchTrainer(
    ...     train_dataloader=train_dataloader,
    ...     val_dataloader=val_dataloader,
    ...     optimizer=optim.Adam(model.parameters(), lr=1e-3),
    ...     max_epochs=10,
    ...     callbacks=[
    ...         EvaluationCallback(my_evaluator),
    ...         EarlyStoppingCallback(patience=3)
    ...     ]
    ... )
    >>>
    >>> # Train model
    >>> state = trainer.train(model, train_dataset, val_dataset)

"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Generic, Literal, cast

import torch
from colt import Lazy
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from formed.workflow import use_step_logger

from ..distributors import BaseDistributor, SingleDeviceDistributor
from ..model import BaseTorchModel
from ..types import IDataLoader, IEvaluator, IOptimizer, ItemT, ModelInputT, ModelOutputT, ModelParamsT
from .callbacks import TorchTrainingCallback
from .engine import DefaultTorchTrainingEngine, TorchTrainingEngine
from .exceptions import StopEarly
from .state import TrainState


class TorchTrainer(
    Generic[
        ItemT,
        ModelInputT,
        ModelOutputT,
        ModelParamsT,
    ]
):
    """High-level trainer for PyTorch models.

    TorchTrainer provides a complete training loop with support for
    distributed training, callbacks, evaluation, and metric logging.
    It handles the coordination of data loading, model training,
    evaluation, and callback execution.

    Type Parameters:
        ItemT: Type of raw dataset items.
        ModelInputT: Type of batched model inputs.
        ModelOutputT: Type of model outputs.
        ModelParamsT: Type of additional model parameters.

    Args:
        train_dataloader: Data loader for training dataset.
        val_dataloader: Optional data loader for validation dataset.
        engine: Training engine (defaults to DefaultTorchTrainingEngine).
        optimizer: PyTorch optimizer.
        callbacks: Sequence of training callbacks.
        distributor: Device distributor (defaults to SingleDeviceDistributor).
        max_epochs: Maximum number of training epochs.
        eval_strategy: When to evaluate - "epoch" or "step".
        eval_interval: Evaluation interval (epochs or steps).
        logging_strategy: When to log - "epoch" or "step".
        logging_interval: Logging interval (epochs or steps).
        logging_first_step: Whether to log after the first training step.

    Example:
        >>> trainer = TorchTrainer(
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     optimizer=torch.optim.Adam(model.parameters(), 1e-3),
        ...     max_epochs=10,
        ...     eval_strategy="epoch",
        ...     logging_strategy="step",
        ...     logging_interval=100
        ... )

    """

    def __init__(
        self,
        *,
        train_dataloader: IDataLoader[ItemT, ModelInputT],
        val_dataloader: IDataLoader[ItemT, ModelInputT] | None = None,
        engine: TorchTrainingEngine[ModelInputT, ModelOutputT, ModelParamsT] | None = None,
        optimizer: Lazy[IOptimizer] | IOptimizer | Callable[[Iterator[torch.nn.Parameter]], IOptimizer],
        callbacks: Sequence[TorchTrainingCallback] = (),
        distributor: BaseDistributor | None = None,
        max_epochs: int = 10,
        eval_strategy: Literal["epoch", "step"] = "epoch",
        eval_interval: int = 1,
        logging_strategy: Literal["epoch", "step"] = "epoch",
        logging_interval: int = 1,
        logging_first_step: bool = True,
    ) -> None:
        self._optimizer = optimizer
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._engine = engine or DefaultTorchTrainingEngine[ModelInputT, ModelOutputT, ModelParamsT]()
        self._distributor = distributor or SingleDeviceDistributor()
        self._max_epochs = max_epochs
        self._eval_strategy = eval_strategy
        self._eval_interval = eval_interval
        self._logging_strategy = logging_strategy
        self._logging_interval = logging_interval
        self._logging_first_step = logging_first_step
        self._callbacks = callbacks

    @property
    def optimizer(self) -> IOptimizer:
        if not isinstance(self._optimizer, IOptimizer):
            raise ValueError("Optimizer has not been initialized. Call train() first.")
        return self._optimizer

    @property
    def distributor(self) -> BaseDistributor:
        return self._distributor

    def train(
        self,
        model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
        train_dataset: Sequence[ItemT],
        val_dataset: Sequence[ItemT] | None = None,
        state: TrainState | None = None,
    ) -> TrainState:
        """Train a model on the provided datasets.

        Args:
            model: Model to train.
            train_dataset: Sequence of training items.
            val_dataset: Optional sequence of validation items.
            state: Optional pre-initialized training state (for resuming).

        Returns:
            Final training state with trained parameters.

        Raises:
            ValueError: If val_dataset is provided but val_dataloader is not.

        Example:
            >>> state = trainer.train(
            ...     model, train_items, val_items
            ... )
            >>> # Load trained parameters
            >>> model.load_state_dict(state.model_state)

        """
        if val_dataset is not None and self._val_dataloader is None:
            raise ValueError("Validation dataloader is not provided.")

        logger = use_step_logger(__name__)

        # Move model to device and wrap if needed
        model = model.to(self._distributor.device)
        model = cast(
            BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
            self._distributor.wrap_model(model),
        )

        if isinstance(self._optimizer, Lazy):
            self._optimizer = self._optimizer.construct(params=model.parameters())
        elif callable(self._optimizer):
            self._optimizer = self._optimizer(model.parameters())

        if state is None:
            state = self._engine.create_state(self, model)

        for callback in self._callbacks:
            callback.on_training_start(self, model, state)

        def get_total_training_steps() -> int:
            dataloader = self._train_dataloader(train_dataset)
            return len(dataloader) * self._max_epochs

        def get_total_eval_steps() -> int:
            assert val_dataset is not None and self._val_dataloader is not None
            dataloader = self._val_dataloader(val_dataset)
            return len(dataloader)

        def new_epoch(epoch: int) -> None:
            assert state is not None
            if self._distributor.is_main_process:
                logger.info(f"Starting epoch {epoch}/{self._max_epochs}")
            for callback in self._callbacks:
                callback.on_epoch_start(self, model, state, epoch)

        def finalize_epoch(epoch: int) -> None:
            assert state is not None
            for callback in self._callbacks:
                callback.on_epoch_end(self, model, state, epoch)

        def new_batch(epoch: int) -> None:
            assert state is not None
            for callback in self._callbacks:
                callback.on_batch_start(self, model, state, epoch)

        def finalize_batch(epoch: int, output: ModelOutputT) -> None:
            assert state is not None
            for callback in self._callbacks:
                callback.on_batch_end(self, model, state, epoch, output)

        def new_evaluators() -> list[IEvaluator[ModelInputT, ModelOutputT]]:
            assert state is not None
            return [callback.on_eval_start(self, model, state) for callback in self._callbacks]

        def update_metrics(
            evaluators: list[IEvaluator[ModelInputT, ModelOutputT]],
            inputs: ModelInputT,
            output: ModelOutputT,
        ) -> None:
            assert state is not None
            for evaluator in evaluators:
                evaluator.update(inputs, output)

        def compute_metrics(evaluators: list[IEvaluator[ModelInputT, ModelOutputT]]) -> dict[str, float]:
            assert state is not None
            metrics = {}
            for evaluator in evaluators:
                metrics.update(evaluator.compute())
            return metrics

        def finalize_evaluation(
            metrics: Mapping[str, float],
        ) -> None:
            assert state is not None
            for callback in self._callbacks:
                callback.on_eval_end(self, model, state, metrics)

        def log(
            metrics: Mapping[str, float],
            prefix: str,
        ) -> None:
            assert state is not None
            if not metrics:
                return
            if self._distributor.is_main_process:
                logger.info("%s", ", ".join(f"{prefix}{k}={v:.4f}" for k, v in metrics.items()))
            for callback in self._callbacks:
                callback.on_log(self, model, state, metrics, prefix=prefix)

        def move_to_device(inputs: ModelInputT) -> ModelInputT:
            """Move inputs to the appropriate device."""
            from typing import Any

            import numpy as np

            visited: set[int] = set()

            def _move(obj: Any) -> Any:
                # Handle tensors
                if isinstance(obj, torch.Tensor):
                    return obj.to(self._distributor.device)

                # Handle numpy arrays - convert to tensor and move to device
                if isinstance(obj, np.ndarray):
                    tensor = torch.from_numpy(obj)
                    # Convert to float32 if it's a floating point tensor
                    if tensor.dtype == torch.float64:
                        tensor = tensor.float()
                    return tensor.to(self._distributor.device)

                # Handle primitives and None
                if obj is None or isinstance(obj, (int, float, str, bool, type)):
                    return obj

                # Check if already visited to avoid infinite recursion
                obj_id = id(obj)
                if obj_id in visited:
                    return obj
                visited.add(obj_id)

                # Handle dict
                if isinstance(obj, dict):
                    return {k: _move(v) for k, v in obj.items()}

                # Handle list/tuple
                if isinstance(obj, (list, tuple)):
                    return type(obj)(_move(x) for x in obj)

                # Handle objects with __dict__ (but not built-in types)
                if hasattr(obj, "__dict__") and not isinstance(obj, type):
                    try:
                        for key, value in list(obj.__dict__.items()):
                            # Skip dunder attributes
                            if not key.startswith("__"):
                                setattr(obj, key, _move(value))
                    except (TypeError, AttributeError):
                        # Skip objects that don't allow attribute modification
                        pass
                    return obj

                return obj

            return cast(ModelInputT, _move(inputs))

        def do_evaluation(progress: Progress) -> None:
            if not val_dataset:
                return

            assert state is not None
            assert self._val_dataloader is not None

            evaluators = new_evaluators()

            task = progress.add_task("Evaluation", total=get_total_eval_steps())
            for batch in self._val_dataloader(val_dataset):
                batch = move_to_device(batch)
                output = self._engine.eval_step(batch, state, self)
                update_metrics(evaluators, batch, output)
                progress.advance(task)
            progress.remove_task(task)

            computed_metrics = compute_metrics(evaluators)
            log(computed_metrics, prefix="val/")
            finalize_evaluation(computed_metrics)

        def is_logging_step(step: int) -> bool:
            return (self._logging_strategy == "step" and step % self._logging_interval == 0) or (
                self._logging_first_step and step == 1
            )

        def is_logging_epoch(epoch: int) -> bool:
            return self._logging_strategy == "epoch" and epoch % self._logging_interval == 0

        def is_eval_step(step: int) -> bool:
            return self._eval_strategy == "step" and step % self._eval_interval == 0

        def is_eval_eopch(epoch: int) -> bool:
            return self._logging_strategy == "epoch" and epoch % self._logging_interval == 0

        evaluators = new_evaluators()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Training", total=get_total_training_steps())
                for epoch in range(1, self._max_epochs + 1):
                    assert state is not None
                    new_epoch(epoch)

                    for batch in self._train_dataloader(train_dataset):
                        new_batch(epoch)

                        batch = move_to_device(batch)
                        output = self._engine.train_step(batch, state, self)

                        update_metrics(evaluators, batch, output)

                        if is_logging_step(int(state.step)):
                            train_metrics = compute_metrics(evaluators)
                            log(train_metrics, prefix="train/")
                            finalize_evaluation(train_metrics)
                            evaluators = new_evaluators()

                        finalize_batch(epoch, output)

                        progress.advance(task)

                        if is_eval_step(int(state.step)):
                            do_evaluation(progress)

                    if is_logging_epoch(epoch):
                        train_metrics = compute_metrics(evaluators)
                        log(train_metrics, prefix="train/")
                        finalize_evaluation(train_metrics)
                        evaluators = new_evaluators()

                    if is_eval_eopch(epoch):
                        do_evaluation(progress)

                    finalize_epoch(epoch)
        except StopEarly:
            assert state is not None
            if self._distributor.is_main_process:
                logger.info(f"Training stopped early at {state.step} steps.")

        for callback in self._callbacks:
            state = callback.on_training_end(self, model, state)

        # Cleanup distributor resources (e.g., DDP process group)
        self._distributor.cleanup()

        return state
