import random
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, TypeVar, Union, overload

from formed.workflow import Format, MappingFormat, step

from .datamodule import DataModule
from .dataset import Dataset

try:
    from formed.integrations.datasets.types import Dataset as HfDataset
except ImportError:

    class HfDataset(Sequence): ...  # type: ignore[no-redef]


T = TypeVar("T")


class DatasetFormat(Format[Dataset]):
    def write(self, artifact: Dataset, directory: Path) -> None:
        shutil.copytree(artifact.path, directory)

    def read(self, directory: Path) -> Dataset:
        return Dataset.from_path(directory)


@overload
def build_datamodule(dataset: HfDataset, datamodule: Optional[DataModule[T]] = ...) -> DataModule: ...


@overload
def build_datamodule(dataset: Sequence[T], datamodule: Optional[DataModule[T]] = ...) -> DataModule[T]: ...


@step("formedml::build_datamodule", format="pickle")
def build_datamodule(
    dataset: Union[Sequence[T], HfDataset],
    datamodule: Optional[DataModule[T]] = None,
) -> DataModule[T]:
    datamodule = datamodule or DataModule()
    datamodule.build(dataset)
    return datamodule


@overload
def split_dataset(dataset: HfDataset, splits: Mapping[str, float], seed: int = ...) -> dict[str, Dataset]: ...


@overload
def split_dataset(dataset: Sequence[T], splits: Mapping[str, float], seed: int = ...) -> dict[str, Dataset[T]]: ...


@step("formedml::split_dataset", format=MappingFormat(DatasetFormat()))
def split_dataset(
    dataset: Union[Sequence[T], HfDataset],
    splits: Mapping[str, float],
    seed: int = 0,
) -> dict[str, Union[Dataset[T], Dataset]]:
    assert abs(sum(splits.values()) - 1.0) < 1e-5, "splits must sum to 1.0"
    rng = random.Random(seed)

    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    splitted_datasets: dict[str, Dataset] = {}

    offset = 0
    for split, ratio in splits.items():
        start_index = offset
        end_index = offset + int(len(dataset) * ratio)
        subset = Dataset.from_iterable(dataset[indices[i]] for i in range(start_index, end_index))
        splitted_datasets[split] = subset
        offset = end_index
    for index in range(offset, len(dataset)):
        splitted_datasets[split].append(dataset[indices[index]])

    return splitted_datasets
