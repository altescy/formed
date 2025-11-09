from collections.abc import Mapping, Sequence
from functools import partial
from typing import Generic, Literal, Optional, Union

import optax
from flax import nnx
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from formed.workflow import use_step_logger

from ..distributors import BaseDistributor, SingleDeviceDistributor
from ..model import BaseFlaxModel
from ..types import IDataLoader, IEvaluator, IOptimizer, ItemT, ModelInputT, ModelOutputT, ModelParamsT
from .callbacks import FlaxTrainingCallback
from .engine import DefaultFlaxTrainingEngine, FlaxTrainingEngine
from .exceptions import StopEarly
from .state import TrainState


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
        *,
        train_dataloader: IDataLoader[ItemT, ModelInputT],
        val_dataloader: Optional[IDataLoader[ItemT, ModelInputT]] = None,
        engine: Optional[FlaxTrainingEngine[ModelInputT, ModelOutputT, ModelParamsT]] = None,
        optimizer: Union[IOptimizer, optax.MultiSteps, optax.GradientTransformation] = optax.adamw(1e-3),
        callbacks: Sequence[FlaxTrainingCallback] = (),
        distributor: Optional[BaseDistributor] = None,
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
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._engine = engine or DefaultFlaxTrainingEngine[ModelInputT, ModelOutputT, ModelParamsT]()
        self._distributor = distributor or SingleDeviceDistributor()
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

    @property
    def distributor(self) -> BaseDistributor:
        return self._distributor

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

        train_step = self._distributor.map(self._engine.train_step, static_argnums=(2,))
        eval_step = partial(nnx.jit, static_argnames=("trainer"))(self._engine.eval_step)

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
            logger.info("%s", ", ".join(f"{prefix}{k}={v:.4f}" for k, v in metrics.items()))
            for callback in self._callbacks:
                callback.on_log(self, model, state, metrics, prefix=prefix)

        def do_evaluation(progress: Progress) -> None:
            if not val_dataset:
                return

            assert state is not None
            assert self._val_dataloader is not None

            evaluators = new_evaluators()

            task = progress.add_task("Evaluation", total=get_total_eval_steps())
            for batch in self._val_dataloader(val_dataset):
                output = eval_step(batch, state, self)
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

                        sharded_batch = self._distributor.shard(batch)
                        replicated_state = self._distributor.replicate(state)

                        replicated_state, replicated_output = train_step(sharded_batch, replicated_state, self)

                        state = self._distributor.unreplicate(replicated_state)
                        output = self._distributor.unreplicate(replicated_output)
                        assert state is not None

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
            logger.info(f"Training stopped early at {state.step} steps.")

        for callback in self._callbacks:
            state = callback.on_training_end(self, model, state)

        return state
