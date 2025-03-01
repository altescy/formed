import json
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Generic, Literal, Optional, TypeVar, Union, cast

import jax
import optax
from colt import Registrable
from flax import nnx
from flax.training import train_state
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from formed.integrations.ml import BasicBatchSampler, DataLoader, MetricAverage
from formed.workflow import use_step_logger, use_step_workdir

from .model import FlaxModel
from .types import DataT, IModelOutput, IOptimizer, ModelInputT, ModelOutputT, ModelParamsT
from .utils import numpy_to_jax

OptimizerT = TypeVar("OptimizerT", bound=optax.GradientTransformation)


class StopEarly(Exception):
    """Raised to stop training early."""


class TrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    graphdef: nnx.GraphDef
    additional_states: tuple[nnx.State, ...] = ()


class FlaxTrainingModule(Registrable, Generic[ModelInputT, ModelOutputT, ModelParamsT]):
    def create_state(
        self,
        rng: jax.random.PRNGKey,
        trainer: "FlaxTrainer",
        model: FlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
    ) -> TrainState:
        raise NotImplementedError

    def train_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "FlaxTrainer",
    ) -> tuple[TrainState, ModelOutputT]:
        raise NotImplementedError

    def eval_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "FlaxTrainer",
    ) -> ModelOutputT:
        raise NotImplementedError


@FlaxTrainingModule.register("default")
class DefaultFlaxTrainingModule(FlaxTrainingModule[ModelInputT, ModelOutputT, ModelParamsT]):
    def create_state(
        self,
        rngs: nnx.Rngs,
        trainer: "FlaxTrainer",
        model: FlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
    ) -> TrainState:
        graphdef, params, *states = nnx.split(model, nnx.Param, nnx.BatchStat, nnx.RngState)
        return cast(
            TrainState,
            TrainState.create(  # type: ignore[no-untyped-call]
                apply_fn=None,
                graphdef=graphdef,
                additional_states=tuple(states),
                params=params,
                tx=trainer.optimizer,
            ),
        )

    @partial(jax.jit, static_argnames=("self", "trainer"))
    def train_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "FlaxTrainer",
    ) -> tuple[TrainState, ModelOutputT]:
        def step(state: TrainState, inputs: ModelInputT) -> tuple[TrainState, ModelOutputT]:
            def loss_fn(params: Any) -> tuple[jax.Array, ModelOutputT]:
                model: FlaxModel[ModelInputT, ModelOutputT, ModelParamsT] = nnx.merge(
                    state.graphdef, params, *state.additional_states
                )  # type: ignore[arg-type]
                output = model(inputs, train=True)
                assert output.loss is not None
                return output.loss, output

            grads, output = jax.grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)  # type: ignore[no-untyped-call]
            return state, output

        return step(state, inputs)

    @partial(jax.jit, static_argnames=("self", "trainer"))
    def eval_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "FlaxTrainer",
    ) -> ModelOutputT:
        model: FlaxModel[ModelInputT, ModelOutputT, ModelParamsT] = nnx.merge(
            state.graphdef, state.params, *state.additional_states  # type: ignore[arg-type]
        )
        return model(inputs, train=False)


class TrainerCallback(Registrable):
    def on_training_start(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
    ) -> None:
        pass

    def on_training_end(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
    ) -> TrainState:
        return state

    def on_epoch_start(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_epoch_end(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_batch_start(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_batch_end(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
        epoch: int,
        output: IModelOutput,
    ) -> None:
        pass

    def on_eval_start(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
    ) -> None:
        pass

    def on_eval_end(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
        metrics: Mapping[str, float],
    ) -> None:
        pass

    def on_log(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
        metrics: Mapping[str, float],
        prefix: str = "",
    ) -> None:
        pass


@TrainerCallback.register("early_stopping")
class EarlyStoppingCallback(TrainerCallback):
    def __init__(
        self,
        patience: int = 5,
        metric: str = "-loss",
    ) -> None:
        import cloudpickle

        self._patience = patience
        self._metric = metric.lstrip("-+")
        self._direction = -1 if metric.startswith("-") else 1
        self._best_metric = -float("inf")
        self._counter = 0
        self._cloudpickle = cloudpickle

    def on_training_start(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
    ) -> None:
        self._best_metric = -float("inf")
        self._counter = 0

    def on_eval_end(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
        metrics: Mapping[str, float],
    ) -> None:
        logger = use_step_logger(__name__)
        workdir = use_step_workdir()
        metric = self._direction * metrics[self._metric]
        if metric > self._best_metric:
            self._best_metric = metric
            self._counter = 0
            with open(workdir / "best_model.pkl", "wb") as file:
                self._cloudpickle.dump(state, file)
            logger.info(f"New best model saved with {self._metric}={self._best_metric:.4f}")
        else:
            self._counter += 1
            if self._counter >= self._patience:
                raise StopEarly()

    def on_training_end(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
    ) -> TrainState:
        logger = use_step_logger(__name__)
        workdir = use_step_workdir()
        if (workdir / "best_model.pkl").exists():
            logger.info("Loading best model.")
            with open(workdir / "best_model.pkl", "rb") as file:
                return cast(TrainState, self._cloudpickle.load(file))
        return state


@TrainerCallback.register("logging")
class LoggingCallback(TrainerCallback):
    def __init__(self) -> None:
        from formed.integrations.mlflow.workflow import MlflowLogger

        self._mlflow_logger: Optional[MlflowLogger] = None

    def on_log(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
        metrics: Mapping[str, float],
        prefix: str = "",
    ) -> None:
        logger = use_step_logger(__name__)
        metrics = {prefix + key: value for key, value in metrics.items()}
        logger.info(json.dumps(metrics, ensure_ascii=False))


@TrainerCallback.register("mlflow")
class MlflowCallback(TrainerCallback):
    def __init__(self) -> None:
        from formed.integrations.mlflow.workflow import MlflowLogger

        self._mlflow_logger: Optional[MlflowLogger] = None

    def on_training_start(
        self,
        trainer: "FlaxTrainer",
        model: FlaxModel,
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
        trainer: "FlaxTrainer",
        model: FlaxModel,
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
        trainer: "FlaxTrainer",
        model: FlaxModel,
        state: TrainState,
        epoch: int,
    ) -> None:
        if self._mlflow_logger is not None:
            self._mlflow_logger.log_metric("epoch", epoch, step=int(state.step))


class FlaxTrainer(
    Generic[
        DataT,
        ModelInputT,
        ModelOutputT,
        ModelParamsT,
    ]
):
    def __init__(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        training_module: Optional[FlaxTrainingModule[ModelInputT, ModelOutputT, ModelParamsT]] = None,
        optimizer: Union[IOptimizer, optax.MultiSteps, optax.GradientTransformation] = optax.adamw(1e-3),
        max_epochs: int = 10,
        eval_strategy: Literal["epoch", "step"] = "epoch",
        eval_interval: int = 1,
        logging_strategy: Literal["epoch", "step"] = "epoch",
        logging_interval: int = 1,
        logging_first_step: bool = True,
        callbacks: Sequence[TrainerCallback] = (),
    ) -> None:
        if not isinstance(optimizer, optax.GradientTransformation):
            optimizer = optax.GradientTransformation(optimizer.init, optimizer.update)

        self._optimizer = optimizer
        self._train_dataloader = train_dataloader or DataLoader(
            BasicBatchSampler(batch_size=32, shuffle=True, drop_last=True)
        )
        self._val_dataloader = val_dataloader or DataLoader(
            BasicBatchSampler(batch_size=32, shuffle=False, drop_last=False)
        )
        self._training_module = training_module
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
        model: FlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        train_dataset: Sequence[DataT],
        val_dataset: Optional[Sequence[DataT]] = None,
        state: Optional[TrainState] = None,
    ) -> TrainState:
        logger = use_step_logger(__name__)

        if self._training_module is None:
            self._training_module = model.default_training_module() or DefaultFlaxTrainingModule()

        if state is None:
            state = self._training_module.create_state(rngs, self, model)

        callbacks = list[TrainerCallback]((*self._callbacks, *model.trainer_callbacks()))
        if not any(isinstance(callback, LoggingCallback) for callback in callbacks):
            callbacks.append(LoggingCallback())

        for callback in callbacks:
            callback.on_training_start(self, model, state)

        def get_total_training_steps() -> int:
            dataloader = self._train_dataloader(train_dataset)
            return len(dataloader) * self._max_epochs

        def get_total_eval_steps() -> int:
            assert val_dataset is not None
            dataloader = self._val_dataloader(val_dataset)
            return len(dataloader)

        def do_evaluation(progress: Progress) -> None:
            assert state is not None
            assert self._training_module is not None

            model.eval()  # type: ignore[no-untyped-call]
            for callback in callbacks:
                callback.on_eval_start(self, model, state)
            if not val_dataset:
                return
            eval_metrics = MetricAverage()

            task = progress.add_task("Evaluation", total=get_total_eval_steps())
            for batch in self._val_dataloader(val_dataset):
                inputs = model.Input(**numpy_to_jax(batch))
                output = self._training_module.eval_step(inputs, state, self)
                assert output.loss is not None
                eval_metrics.add(
                    {
                        "loss": float(output.loss.item()),
                        **{key: float(value.item()) for key, value in (output.metrics or {}).items()},
                    },
                    batch.size,
                )
                progress.advance(task)
            progress.remove_task(task)
            for callback in callbacks:
                callback.on_log(self, model, state, eval_metrics, prefix="val/")
            for callback in callbacks:
                callback.on_eval_end(self, model, state, eval_metrics)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
            ) as progress:
                train_metrics = MetricAverage()
                task = progress.add_task("Training", total=get_total_training_steps())
                for epoch in range(1, self._max_epochs + 1):
                    for callback in callbacks:
                        callback.on_epoch_start(self, model, state, epoch)

                    model.train()  # type: ignore[no-untyped-call]
                    for batch in self._train_dataloader(train_dataset):
                        for callback in callbacks:
                            callback.on_batch_start(self, model, state, epoch)

                        inputs = model.Input(**numpy_to_jax(batch))
                        state, output = self._training_module.train_step(inputs, state, self)

                        assert output.loss is not None
                        train_metrics.add(
                            {
                                "loss": float(output.loss.item()),
                                **{key: float(value.item()) for key, value in (output.metrics or {}).items()},
                            },
                            batch.size,
                        )

                        if (self._logging_strategy == "step" and state.step % self._logging_interval == 0) or (
                            self._logging_first_step and state.step == 1
                        ):
                            for callback in callbacks:
                                callback.on_log(self, model, state, train_metrics, prefix="train/")
                            train_metrics.reset()

                        for callback in callbacks:
                            callback.on_batch_end(self, model, state, epoch, output)

                        progress.advance(task)

                        if self._eval_strategy == "step" and state.step % self._eval_interval == 0:
                            do_evaluation(progress)

                    if self._eval_strategy == "epoch" and epoch % self._eval_interval == 0:
                        do_evaluation(progress)

                    if self._logging_strategy == "epoch" and epoch % self._logging_interval == 0:
                        for callback in callbacks:
                            callback.on_log(self, model, state, train_metrics, prefix="train/")
                        train_metrics.reset()

                    for callback in callbacks:
                        callback.on_epoch_end(self, model, state, epoch)
        except StopEarly:
            logger.info(f"Training stopped early at {state.step} steps.")

        for callback in callbacks:
            state = callback.on_training_end(self, model, state)

        return state
