from collections.abc import Callable, Sequence
from typing import Generic, Literal, Optional, Union

import optax
from flax import nnx
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from formed.integrations.ml import AverageMetric
from formed.workflow import use_step_logger

from ..model import BaseFlaxModel
from ..types import IDataLoader, IOptimizer, ItemT, ModelInputT, ModelOutputT, ModelParamsT
from .callbacks import FlaxTrainingCallback
from .engine import DefaultFlaxTrainingEngine, FlaxTrainingEngine
from .exceptions import StopEarly
from .state import TrainState


def _default_evaluator(output) -> dict[str, float]:
    return {"loss": (getattr(output, "loss").item())}


class FlaxTrainer(
    Generic[
        ItemT,
        ModelInputT,
        ModelOutputT,
        ModelParamsT,
    ]
):
    def __init__(
        self,
        train_dataloader: IDataLoader[ItemT, ModelInputT],
        val_dataloader: Optional[IDataLoader[ItemT, ModelInputT]] = None,
        engine: Optional[FlaxTrainingEngine[ModelInputT, ModelOutputT, ModelParamsT]] = None,
        optimizer: Union[IOptimizer, optax.MultiSteps, optax.GradientTransformation] = optax.adamw(1e-3),
        evaluator: Optional[Callable[[ModelOutputT], dict[str, float]]] = None,
        callbacks: Sequence[FlaxTrainingCallback] = (),
        max_epochs: int = 10,
        eval_strategy: Literal["epoch", "step"] = "epoch",
        eval_interval: int = 1,
        logging_strategy: Literal["epoch", "step"] = "epoch",
        logging_interval: int = 1,
        logging_first_step: bool = True,
    ) -> None:
        if not isinstance(optimizer, optax.GradientTransformation):
            optimizer = optax.GradientTransformation(optimizer.init, optimizer.update)  # pyright: ignore[reportArgumentType]

        self._optimizer = optimizer
        self._evaluator = evaluator or _default_evaluator
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._engine = engine or DefaultFlaxTrainingEngine[ModelInputT, ModelOutputT, ModelParamsT]()
        self._max_epochs = max_epochs
        self._eval_strategy = eval_strategy
        self._eval_interval = eval_interval
        self._logging_strategy = logging_strategy
        self._logging_interval = logging_interval
        self._logging_first_step = logging_first_step
        self._callbacks = callbacks

    @property
    def optimizer(self) -> optax.GradientTransformation:
        return self._optimizer

    def train(
        self,
        rngs: nnx.Rngs,
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        train_dataset: Sequence[ItemT],
        val_dataset: Optional[Sequence[ItemT]] = None,
        state: Optional[TrainState] = None,
    ) -> TrainState:
        if val_dataset is not None and self._val_dataloader is None:
            raise ValueError("Validation dataloader is not provided.")

        logger = use_step_logger(__name__)

        if state is None:
            state = self._engine.create_state(rngs, self, model)

        callbacks = self._callbacks

        for callback in callbacks:
            callback.on_training_start(self, model, state)

        def get_total_training_steps() -> int:
            dataloader = self._train_dataloader(train_dataset)
            return len(dataloader) * self._max_epochs

        def get_total_eval_steps() -> int:
            assert val_dataset is not None and self._val_dataloader is not None
            dataloader = self._val_dataloader(val_dataset)
            return len(dataloader)

        def do_evaluation(progress: Progress) -> None:
            assert state is not None
            assert val_dataset is not None
            assert self._val_dataloader is not None

            model.eval()
            for callback in callbacks:
                callback.on_eval_start(self, model, state)
            if not val_dataset:
                return
            metrics = AverageMetric()

            task = progress.add_task("Evaluation", total=get_total_eval_steps())
            for batch in self._val_dataloader(val_dataset):
                output = self._engine.eval_step(batch, state, self)
                metrics.update(self._evaluator(output))
                progress.advance(task)
            progress.remove_task(task)
            computed_metrics = metrics.compute()
            for callback in callbacks:
                callback.on_log(self, model, state, computed_metrics, prefix="val/")
            for callback in callbacks:
                callback.on_eval_end(self, model, state, computed_metrics)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
            ) as progress:
                train_metrics = AverageMetric()
                task = progress.add_task("Training", total=get_total_training_steps())
                for epoch in range(1, self._max_epochs + 1):
                    assert state is not None
                    for callback in callbacks:
                        callback.on_epoch_start(self, model, state, epoch)

                    model.train()
                    for batch in self._train_dataloader(train_dataset):
                        for callback in callbacks:
                            callback.on_batch_start(self, model, state, epoch)

                        state, output = self._engine.train_step(batch, state, self)
                        assert state is not None

                        train_metrics.update(self._evaluator(output))
                        computed_train_metrics = train_metrics.compute()

                        if (self._logging_strategy == "step" and state.step % self._logging_interval == 0) or (
                            self._logging_first_step and state.step == 1
                        ):
                            for callback in callbacks:
                                callback.on_log(self, model, state, computed_train_metrics, prefix="train/")
                            train_metrics.reset()

                        for callback in callbacks:
                            callback.on_batch_end(self, model, state, epoch, output)

                        progress.advance(task)

                        if self._eval_strategy == "step" and state.step % self._eval_interval == 0:
                            do_evaluation(progress)

                    if self._eval_strategy == "epoch" and epoch % self._eval_interval == 0:
                        do_evaluation(progress)

                    if self._logging_strategy == "epoch" and epoch % self._logging_interval == 0:
                        computed_train_metrics = train_metrics.compute()
                        for callback in callbacks:
                            callback.on_log(self, model, state, computed_train_metrics, prefix="train/")
                        train_metrics.reset()

                    for callback in callbacks:
                        callback.on_epoch_end(self, model, state, epoch)
        except StopEarly:
            assert state is not None
            logger.info(f"Training stopped early at {state.step} steps.")

        for callback in callbacks:
            state = callback.on_training_end(self, model, state)

        return state
