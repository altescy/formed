import random
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, TypeVar

from formed.workflow import Format, MappingFormat, step

from .datamodule import DataModule
from .dataset import Dataset

T = TypeVar("T")


class DatasetFormat(Format[Dataset]):
    def write(self, artifact: Dataset, directory: Path) -> None:
        shutil.copytree(artifact.path, directory)

    def read(self, directory: Path) -> Dataset:
        return Dataset.from_path(directory)


@step("formedml::build_datamodule", format="pickle")
def build_datamodule(
    dataset: Sequence[T],
    datamodule: Optional[DataModule[T]] = None,
) -> DataModule[T]:
    datamodule = datamodule or DataModule()
    datamodule.build(dataset)
    return datamodule


@step("formedml::split_dataset", format=MappingFormat(DatasetFormat()))
def split_dataset(
    dataset: Sequence[T],
    splits: Mapping[str, float],
    seed: int = 0,
) -> dict[str, Dataset[T]]:
    assert abs(sum(splits.values()) - 1.0) < 1e-5, "splits must sum to 1.0"
    rng = random.Random(seed)

    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    split_datasets: dict[str, Dataset] = {}

    offset = 0
    for split, ratio in splits.items():
        start_index = offset
        end_index = offset + int(len(dataset) * ratio)
        subset = Dataset.from_iterable(dataset[indices[i]] for i in range(start_index, end_index))
        split_datasets[split] = subset
        offset = end_index
    if split_datasets:
        last_split = list(splits)[-1]
        for index in range(offset, len(dataset)):
            split_datasets[last_split].append(dataset[indices[index]])

    return split_datasets
