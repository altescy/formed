import contextvars
from collections.abc import Sequence
from typing import Optional, TypeVar, Union, cast

from colt import Lazy
from flax import nnx

from formed.integrations.ml import DataModule, Dataset
from formed.workflow import step, use_step_logger

from .model import FlaxModel
from .training import FlaxTrainer, TrainState
from .types import DataT, ModelInputT, ModelOutputT, ModelParamsT

T = TypeVar("T")


@step("flax::train", format="cloudpickle")
def train_flax_model(
    model: Lazy[FlaxModel[ModelInputT, ModelOutputT, ModelParamsT]],
    trainer: FlaxTrainer[DataT, ModelInputT, ModelOutputT, ModelParamsT],
    train_dataset: Union[Sequence[T], Sequence[DataT]],
    val_dataset: Optional[Union[Sequence[T], Sequence[DataT]]] = None,
    datamodule: Optional[DataModule[T]] = None,
    seed: int = 0,
) -> FlaxModel:
    logger = use_step_logger(__name__)

    rngs = nnx.Rngs(seed)

    if datamodule is None:
        if isinstance(model.constructor, type) and issubclass(model.constructor, FlaxModel):
            datamodule = model.constructor.default_data_module()
        if datamodule is None:
            datamodule = DataModule()
        logger.info(f"Using default DataModule: {datamodule}")

    if datamodule is not None:
        logger.info("Building DataModule...")
        train_dataset = cast(Sequence[T], train_dataset)
        datamodule.build(train_dataset)
        train_dataset = cast(Sequence[DataT], Dataset.from_iterable(datamodule(train_dataset)))
        if val_dataset is not None:
            val_dataset = cast(Sequence[T], val_dataset)
            val_dataset = cast(Sequence[DataT], Dataset.from_iterable(datamodule(val_dataset)))

    def train() -> TrainState:
        logger.info("Training model...")
        if datamodule is not None:
            datamodule.activate()
        return trainer.train(
            rngs=rngs,
            model=model.construct(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

    ctx = contextvars.copy_context()
    state = ctx.run(train)

    return nnx.merge(state.graphdef, state.params, *state.additional_states)  # type: ignore[arg-type]
