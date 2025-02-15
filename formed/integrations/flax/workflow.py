from collections.abc import Sequence
from typing import Optional

from flax import nnx

from formed.workflow import step

from .model import FlaxModel
from .training import FlaxTrainer
from .types import DataT, ModelInputT, ModelOutputT, ModelParamsT


@step("flax::train", format="cloudpickle")
def train_flax_model(
    model: FlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
    trainer: FlaxTrainer[DataT, ModelInputT, ModelOutputT, ModelParamsT],
    train_dataset: Sequence[DataT],
    val_dataset: Optional[Sequence[DataT]] = None,
    seed: int = 0,
) -> FlaxModel:
    rngs = nnx.Rngs(seed)

    state = trainer.train(
        rngs=rngs,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    return nnx.merge(state.graphdef, state.params, *state.additional_states)  # type: ignore[arg-type]
