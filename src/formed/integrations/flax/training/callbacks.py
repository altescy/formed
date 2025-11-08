from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, cast

import cloudpickle
from colt import Registrable

from formed.workflow import use_step_logger, use_step_workdir

from ..model import BaseFlaxModel
from ..types import ItemT, ModelInputT, ModelOutputT, ModelParamsT
from .exceptions import StopEarly
from .state import TrainState

if TYPE_CHECKING:
    from .trainer import FlaxTrainer


class FlaxTrainingCallback(Registrable):
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
    ) -> None:
        pass

    def on_eval_end(
        self,
        trainer: "FlaxTrainer[ItemT, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
        state: TrainState,
        metrics: Mapping[str, float],
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


@FlaxTrainingCallback.register("early_stopping")
class EarlyStoppingCallback(FlaxTrainingCallback):
    def __init__(
        self,
        patience: int = 5,
        metric: str = "-loss",
    ) -> None:
        self._patience = patience
        self._metric = metric.lstrip("-+")
        self._direction = -1 if metric.startswith("-") else 1
        self._best_metric = -float("inf")
        self._counter = 0

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
    ) -> None:
        logger = use_step_logger(__name__)
        workdir = use_step_workdir()
        metric = self._direction * metrics[self._metric]
        if metric > self._best_metric:
            self._best_metric = metric
            self._counter = 0
            with open(workdir / "best_model.pkl", "wb") as file:
                cloudpickle.dump(state, file)
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
        workdir = use_step_workdir()
        if (workdir / "best_model.pkl").exists():
            logger.info("Loading best model.")
            with open(workdir / "best_model.pkl", "rb") as file:
                return cast(TrainState, cloudpickle.load(file))
        return state


@FlaxTrainingCallback.register("mlflow")
class MlflowCallback(FlaxTrainingCallback):
    def __init__(self) -> None:
        from formed.integrations.mlflow.workflow import MlflowLogger

        self._mlflow_logger: Optional[MlflowLogger] = None

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
