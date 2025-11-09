from collections.abc import Sequence
from typing import Optional

from flax import nnx

from formed.workflow import step

from .model import BaseFlaxModel
from .training import FlaxTrainer
from .types import ItemT


@step("flax::train")
def train_flax_model(
    model: BaseFlaxModel,
    trainer: FlaxTrainer,
    train_dataset: Sequence[ItemT],
    val_dataset: Optional[Sequence[ItemT]] = None,
    random_seed: int = 0,
) -> BaseFlaxModel:
    rngs = nnx.Rngs(random_seed)
    state = trainer.train(rngs, model, train_dataset, val_dataset)
    return nnx.merge(state.graphdef, state.params, *state.additional_states)
