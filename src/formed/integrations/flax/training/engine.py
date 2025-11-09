import abc
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, Union, cast

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
    @abc.abstractmethod
    def create_state(
        self,
        rngs: nnx.Rngs,
        trainer: "FlaxTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
        model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
    ) -> TrainState:
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "FlaxTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
    ) -> tuple[TrainState, ModelOutputT]:
        raise NotImplementedError

    @abc.abstractmethod
    def eval_step(
        self,
        inputs: ModelInputT,
        state: TrainState,
        trainer: "FlaxTrainer[Any, ModelInputT, ModelOutputT, ModelParamsT]",
    ) -> ModelOutputT:
        raise NotImplementedError


@FlaxTrainingEngine.register("default")
class DefaultFlaxTrainingEngine(FlaxTrainingEngine[ModelInputT, ModelOutputT, ModelParamsT]):
    def __init__(self, loss: Union[str, Callable[[ModelOutputT], jax.Array]] = "loss") -> None:
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
        del trainer

        def step(state: TrainState, inputs: ModelInputT) -> tuple[TrainState, ModelOutputT]:
            def loss_fn(params: Any) -> tuple[jax.Array, ModelOutputT]:
                model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT] = nnx.merge(
                    state.graphdef, params, *state.additional_states
                )
                model.train()
                output = model(inputs)
                loss = self._loss(output)
                return loss, output

            grads, output = jax.grad(loss_fn, has_aux=True)(state.params)
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
