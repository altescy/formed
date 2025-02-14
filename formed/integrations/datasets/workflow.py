from collections.abc import Mapping
from contextlib import suppress
from os import PathLike
from pathlib import Path
from typing import Any, Generic, Union, cast

import datasets
import minato

from formed.workflow import Format, step, use_step_logger

from .types import Dataset, DatasetOrMappingT


@Format.register("datasets")
class DatasetFormat(Generic[DatasetOrMappingT], Format[DatasetOrMappingT]):
    def write(self, artifact: DatasetOrMappingT, directory: Path) -> None:
        if isinstance(artifact, Mapping):
            for key, dataset in artifact.items():
                dataset.save_to_disk(str(directory / f"data.{key}"))
        else:
            artifact.save_to_disk(str(directory / "data"))

    def read(self, directory: Path) -> DatasetOrMappingT:
        if (directory / "data").exists():
            return cast(DatasetOrMappingT, datasets.load_from_disk(str(directory / "data")))
        else:
            return cast(
                DatasetOrMappingT,
                {datadir.name[5:]: datasets.load_from_disk(str(datadir)) for datadir in directory.glob("data.*")},
            )


@step("datasets::load_dataset", cacheable=False, format=DatasetFormat())
def load_dataset(
    path: Union[str, PathLike],
    **kwargs: Any,
) -> Dataset:
    with suppress(FileNotFoundError):
        path = minato.cached_path(path)
    if Path(path).exists():
        dataset = datasets.load_from_disk(str(path))
    else:
        dataset = cast(Dataset, datasets.load_dataset(str(path), **kwargs))
    if not isinstance(dataset, (datasets.Dataset, datasets.DatasetDict)):
        raise ValueError("Only Dataset or DatasetDict is supported")
    return dataset


@step("datasets::compose_datasetdict", format=DatasetFormat())
def compose_datasetdict(**kwargs: Dataset) -> datasets.DatasetDict:
    datasets_: dict[str, datasets.Dataset] = {
        key: dataset for key, dataset in kwargs.items() if isinstance(dataset, datasets.Dataset)
    }
    if len(datasets_) != len(kwargs):
        logger = use_step_logger(__name__)
        logger.warning(
            "Following keys are ignored since they are not Dataset instances: %s",
            set(kwargs) - set(datasets_),
        )
    return datasets.DatasetDict(datasets_)


@step("datasets::concatenate_datasets", format=DatasetFormat())
def concatenate_datasets(dsets: list[datasets.Dataset], **kwargs: Any) -> datasets.Dataset:
    return cast(datasets.Dataset, datasets.concatenate_datasets(dsets, **kwargs))


@step("datasets::train_test_split", format=DatasetFormat())
def train_test_split(
    dataset: Dataset,
    train_key: str = "train",
    test_key: str = "test",
    **kwargs: Any,
) -> dict[str, Dataset]:
    if isinstance(dataset, datasets.Dataset):
        splitted = dataset.train_test_split(**kwargs)
        return {train_key: splitted["train"], test_key: splitted["test"]}
    else:
        train_datasets: dict[str, datasets.Dataset] = {}
        test_datasets: dict[str, datasets.Dataset] = {}
        for key, dset in dataset.items():
            splitted = dset.train_test_split(**kwargs)
            train_datasets[key] = splitted["train"]
            test_datasets[key] = splitted["test"]
        return {
            train_key: datasets.DatasetDict(train_datasets),
            test_key: datasets.DatasetDict(test_datasets),
        }
