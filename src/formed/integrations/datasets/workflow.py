from collections.abc import Mapping
from contextlib import suppress
from os import PathLike
from pathlib import Path
from typing import Any, Generic, cast

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
        return cast(
            DatasetOrMappingT,
            {datadir.name[5:]: datasets.load_from_disk(str(datadir)) for datadir in directory.glob("data.*")},
        )


@step("datasets::load", cacheable=False, format=DatasetFormat())
def load_dataset(
    path: str | PathLike,
    **kwargs: Any,
) -> Dataset:
    """Load a dataset from disk or the Hugging Face Hub.

    This step loads a dataset from a local path or downloads it from the
    Hugging Face Hub. The dataset can be either a Dataset or DatasetDict.

    Args:
        path: Path to the dataset (local or remote).
        **kwargs: Additional arguments to pass to `datasets.load_dataset`.

    Returns:
        Loaded Dataset or DatasetDict.

    Raises:
        ValueError: If the loaded object is not a Dataset or DatasetDict.
    """
    with suppress(FileNotFoundError):
        path = minato.cached_path(path)
    if Path(path).exists():
        dataset = datasets.load_from_disk(str(path))
    else:
        dataset = cast(Dataset, datasets.load_dataset(str(path), **kwargs))
    if not isinstance(dataset, (datasets.Dataset, datasets.DatasetDict)):
        raise ValueError("Only Dataset or DatasetDict is supported")
    return dataset


@step("datasets::compose", format=DatasetFormat())
def compose_datasetdict(**kwargs: Dataset) -> datasets.DatasetDict:
    """Compose multiple Dataset objects into a single DatasetDict.

    This step combines individual Dataset objects into a DatasetDict,
    filtering out any non-Dataset values.

    Args:
        **kwargs: Named datasets to compose. Only Dataset instances are included.

    Returns:
        DatasetDict containing all provided Dataset instances.
    """
    datasets_: dict[str, datasets.Dataset] = {
        key: dataset for key, dataset in kwargs.items() if isinstance(dataset, datasets.Dataset)
    }
    if len(datasets_) != len(kwargs):
        logger = use_step_logger(__name__)
        logger.warning(
            "Following keys are ignored since they are not Dataset instances: %s",
            set(kwargs) - set(datasets_),
        )
    return datasets.DatasetDict(**datasets_)


@step("datasets::concatenate", format=DatasetFormat())
def concatenate_datasets(dsets: list[datasets.Dataset], **kwargs: Any) -> datasets.Dataset:
    """Concatenate multiple datasets into a single dataset.

    Args:
        dsets: List of datasets to concatenate.
        **kwargs: Additional arguments to pass to `datasets.concatenate_datasets`.

    Returns:
        Concatenated dataset.
    """
    return cast(datasets.Dataset, datasets.concatenate_datasets(dsets, **kwargs))


@step("datasets::train_test_split", format=DatasetFormat())
def train_test_split(
    dataset: Dataset,
    train_key: str = "train",
    test_key: str = "test",
    **kwargs: Any,
) -> dict[str, Dataset]:
    """Split a dataset into train and test sets.

    This step splits a Dataset or DatasetDict into training and test sets.
    For DatasetDict inputs, each split is performed independently.

    Args:
        dataset: Dataset or DatasetDict to split.
        train_key: Key name for the training split.
        test_key: Key name for the test split.
        **kwargs: Additional arguments to pass to `train_test_split`.

    Returns:
        Dictionary with train and test splits.
    """
    if isinstance(dataset, datasets.Dataset):
        split = dataset.train_test_split(**kwargs)
        return {train_key: split["train"], test_key: split["test"]}
    else:
        train_datasets: dict[str, datasets.Dataset] = {}
        test_datasets: dict[str, datasets.Dataset] = {}
        for key, dset in dataset.items():
            split = dset.train_test_split(**kwargs)
            train_datasets[str(key)] = split["train"]
            test_datasets[str(key)] = split["test"]
        return {
            train_key: datasets.DatasetDict(**train_datasets),
            test_key: datasets.DatasetDict(**test_datasets),
        }
