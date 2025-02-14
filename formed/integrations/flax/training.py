from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from typing import Generic, Literal, Optional, TypeVar, Union, cast

import flax
import jax
import optax
from colt import Registrable
from flax import nnx
from flax.training import train_state

from formed.workflow import use_step_logger

from .data import DataLoader
from .model import FlaxModel
from .types import IModelOutput, IOptimizer, ModelInputT, ModelOutputT, ModelParamsT

DataT = TypeVar("DataT")
OptimizerT = TypeVar("OptimizerT", bound=optax.GradientTransformation)


class StopEarly(Exception):
    """Raised to stop training early."""


class TrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    graphdef: nnx.GraphDef


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
        graphdef, params = nnx.split(model, nnx.Param)
        return cast(
            TrainState,
            TrainState.create(  # type: ignore[no-untyped-call]
                apply_fn=None,
                graphdef=graphdef,
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

            def loss_fn(params: Mapping) -> tuple[jax.Array, ModelOutputT]:
                model: FlaxModel[ModelInputT, ModelOutputT, ModelParamsT] = nnx.merge(state.graphdef, params)
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
        model: FlaxModel[ModelInputT, ModelOutputT, ModelParamsT] = nnx.merge(state.graphdef, state.params)
        return model(inputs, train=False)


class TrainerCallback(Registrable):
    def on_training_start(
        self,
        trainer: "FlaxTrainer",
        state: TrainState,
    ) -> None:
        pass

    def on_training_end(
        self,
        trainer: "FlaxTrainer",
        state: TrainState,
    ) -> TrainState:
        return state

    def on_epoch_start(
        self,
        trainer: "FlaxTrainer",
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_epoch_end(
        self,
        trainer: "FlaxTrainer",
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_batch_start(
        self,
        trainer: "FlaxTrainer",
        state: TrainState,
        epoch: int,
    ) -> None:
        pass

    def on_batch_end(
        self,
        trainer: "FlaxTrainer",
        state: TrainState,
        epoch: int,
        output: IModelOutput,
    ) -> None:
        pass

    def on_log(
        self,
        trainer: "FlaxTrainer",
        state: TrainState,
        loss: float,
        metrics: Mapping[str, float],
    ) -> None:
        pass


@TrainerCallback.register("mlflow")
class MlflowCallback(TrainerCallback):
    def __init__(self) -> None:
        from formed.integrations.mlflow.workflow import MlflowLogger

        self._mlflow_logger: Optional[MlflowLogger] = None

    def on_training_start(
        self,
        trainer: "FlaxTrainer",
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
        state: TrainState,
        loss: float,
        metrics: Mapping[str, float],
    ) -> None:
        if self._mlflow_logger is not None:
            self._mlflow_logger.log_metric("loss", loss, step=int(state.step))
            for key, value in metrics.items():
                self._mlflow_logger.log_metric(f"metric/{key}", value, step=int(state.step))

    def on_epoch_end(
        self,
        trainer: "FlaxTrainer",
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
        train_dataloader: DataLoader[DataT, ModelInputT],
        val_dataloader: Optional[DataLoader] = None,
        training_module: Optional[FlaxTrainingModule[ModelInputT, ModelOutputT, ModelParamsT]] = None,
        optimizer: Union[IOptimizer, optax.GradientTransformation] = optax.adamw(1e-3),
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
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader or train_dataloader
        self._training_module = training_module or DefaultFlaxTrainingModule()
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
    ) -> TrainState:
        logger = use_step_logger(__name__)

        state = self._training_module.create_state(rngs, self, model)

        for callback in self._callbacks:
            callback.on_training_start(self, state)

        def do_evaluation() -> None:
            model.eval()  # type: ignore[no-untyped-call]
            if not val_dataset:
                return
            losses: list[jax.Array] = []
            eval_metrics: list[Mapping[str, jax.Array]] = []
            for batch in self._val_dataloader(val_dataset):
                output = self._training_module.eval_step(batch, state, self)
                assert output.loss is not None
                losses.append(output.loss)
                eval_metrics.append(output.metrics or {})
            eval_loss = float(jax.numpy.mean(jax.numpy.stack(losses)).item())
            eval_metrics_mean = {
                f"val/{key}": float(value.item())
                for key, value in jax.tree.map(
                    lambda x: x / len(eval_metrics),
                    jax.tree.map(jax.numpy.sum, eval_metrics),
                ).items()
            }
            for callback in self._callbacks:
                callback.on_log(self, state, eval_loss, eval_metrics_mean)

        try:
            for epoch in range(1, self._max_epochs + 1):
                for callback in self._callbacks:
                    callback.on_epoch_start(self, state, epoch)

                model.train()  # type: ignore[no-untyped-call]
                for batch in self._train_dataloader(train_dataset):
                    for callback in self._callbacks:
                        callback.on_batch_start(self, state, epoch)

                    state, output = self._training_module.train_step(batch, state, self)

                    if (self._logging_strategy == "step" and state.step % self._logging_interval == 0) or (
                        self._logging_first_step and state.step == 1
                    ):
                        assert output.loss is not None
                        loss = float(output.loss.item())
                        train_metrics = {
                            f"train/{key}": float(value.item()) for key, value in (output.metrics or {}).items()
                        }
                        for callback in self._callbacks:
                            if loss is not None:
                                callback.on_log(self, state, loss, train_metrics)

                    for callback in self._callbacks:
                        callback.on_batch_end(self, state, epoch, output)

                    if self._eval_strategy == "step" and state.step % self._eval_interval == 0:
                        do_evaluation()

                if self._eval_strategy == "epoch" and epoch % self._eval_interval == 0:
                    do_evaluation()

                if self._logging_strategy == "epoch" and epoch % self._logging_interval == 0:
                    assert output.loss is not None
                    loss = float(output.loss.item())
                    for callback in self._callbacks:
                        callback.on_log(self, state, loss, train_metrics)

                for callback in self._callbacks:
                    callback.on_epoch_end(self, state, epoch)
        except StopEarly:
            logger.info(f"Training stopped early at {state.step} steps.")

        for callback in self._callbacks:
            state = callback.on_training_end(self, state)

        return state
