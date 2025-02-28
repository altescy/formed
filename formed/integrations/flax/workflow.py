import contextvars
from collections.abc import Sequence
from typing import Annotated, Optional, TypeVar, Union, cast

from colt import Lazy
from flax import nnx

from formed.integrations.ml import BasicBatchSampler, DataLoader, DataModule, Dataset, MetricAverage
from formed.integrations.ml.workflow import split_dataset
from formed.workflow import WorkflowStepResultFlag, step, use_step_logger

from .model import FlaxModel
from .training import FlaxTrainer, TrainState
from .types import DataT, ModelInputT, ModelOutputT, ModelParamsT
from .utils import numpy_to_jax

try:
    from formed.integrations.datasets.types import Dataset as HfDataset
except ImportError:
    HfDataset = None  # type: ignore[misc]

T = TypeVar("T")


@step("flax::train", format="cloudpickle")
def train_flax_model(
    model: Lazy[FlaxModel[ModelInputT, ModelOutputT, ModelParamsT]],
    trainer: FlaxTrainer[DataT, ModelInputT, ModelOutputT, ModelParamsT],
    train_dataset: Union[Sequence[T], Sequence[DataT], HfDataset],
    val_dataset: Optional[Union[float, Sequence[T], Sequence[DataT], HfDataset]] = None,
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
            if isinstance(val_dataset, float):
                if val_dataset < 0.0 or val_dataset > 1.0:
                    raise ValueError("val_dataset must be in the range [0.0, 1.0]")
                logger.info("Splitting dataset...")
                splitted_datasets = split_dataset(
                    train_dataset, {"train": 1.0 - val_dataset, "val": val_dataset}, seed=seed
                )
                logger.info(
                    "Splitted dataset sizes: %s", {key: len(dataset) for key, dataset in splitted_datasets.items()}
                )
                train_dataset = splitted_datasets["train"]
                val_dataset = splitted_datasets["val"]
            else:
                val_dataset = cast(Sequence[T], val_dataset)
                val_dataset = cast(Sequence[DataT], Dataset.from_iterable(datamodule(val_dataset)))

    def train() -> TrainState:
        logger.info("Training model...")
        if datamodule is not None:
            datamodule.activate()
        return trainer.train(
            rngs=rngs,
            model=model.construct(),
            train_dataset=cast(Sequence[DataT], train_dataset),
            val_dataset=cast(Optional[Sequence[DataT]], val_dataset),
        )

    ctx = contextvars.copy_context()
    state = ctx.run(train)

    return nnx.merge(state.graphdef, state.params, *state.additional_states)  # type: ignore[arg-type]


@step("flax::evaluate")
def evaluate_flax_model(
    model: FlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
    datamodule: DataModule[T],
    dataset: Union[Sequence[T], Sequence[DataT], HfDataset],
    params: Optional[ModelParamsT] = None,
    dataloader: Optional[DataLoader] = None,
) -> Annotated[dict[str, float], WorkflowStepResultFlag.METRICS]:
    from rich.progress import Progress

    if dataloader is None:
        dataloader = DataLoader(BasicBatchSampler(batch_size=32, drop_last=False, shuffle=False))

    model.eval()  # type: ignore[no-untyped-call]

    metrics = MetricAverage()

    instances = Dataset.from_iterable(datamodule(cast(Sequence, dataset)))
    with Progress() as progress:
        iterator = dataloader(instances)
        task = progress.add_task("Evaluating...", total=len(iterator))
        for batch in iterator:
            inputs = model.Input(**numpy_to_jax(batch))
            output = model(inputs, params)
            metrics.add({key: float(value.item()) for key, value in (output.metrics or {}).items()})
            progress.update(task, advance=1)

    return dict(metrics)
