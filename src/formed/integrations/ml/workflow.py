import dataclasses
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Generic

from typing_extensions import TypeVar

from formed.common.dataset import Dataset
from formed.common.rich import progress
from formed.workflow import step
from formed.workflow.format import Format, JsonFormat, PickleFormat

from .transforms import DataModule
from .types import AsConverter, AsInstance

_InputT = TypeVar("_InputT", default=Any)
_InstanceT = TypeVar("_InstanceT", bound=DataModule[AsInstance], default=Any)


@dataclasses.dataclass(frozen=True)
class DataModuleAndInstances(Generic[_InputT, _InstanceT]):
    datamodule: DataModule[AsConverter, _InputT, _InstanceT]
    instances: Iterable[_InstanceT]


@Format.register("ml::datamodule_and_dataset")
class DataModuleAndInstancesFormat(
    Format[DataModuleAndInstances[_InputT, _InstanceT]],
    Generic[_InputT, _InstanceT],
):
    _INSTANCES_FORMAT = PickleFormat[Iterable[_InstanceT]]()
    _DATAMODULE_FORMAT = JsonFormat[DataModule[AsConverter, _InputT]]()

    def write(
        self,
        artifact: DataModuleAndInstances[_InputT, _InstanceT],
        directory: Path,
    ) -> None:
        instances_path = directory / "instances"
        datamodule_path = directory / "datamodule"

        instances_path.mkdir(parents=True, exist_ok=True)
        datamodule_path.mkdir(parents=True, exist_ok=True)

        self._INSTANCES_FORMAT.write(artifact.instances, instances_path)
        self._DATAMODULE_FORMAT.write(artifact.datamodule, datamodule_path)

    def read(self, directory: Path) -> DataModuleAndInstances[_InputT, _InstanceT]:
        instances_path = directory / "instances"
        datamodule_path = directory / "datamodule"

        instances = self._INSTANCES_FORMAT.read(instances_path)
        datamodule = self._DATAMODULE_FORMAT.read(datamodule_path)

        return DataModuleAndInstances(datamodule=datamodule, instances=instances)


@step("ml::train_datamodule", format="json")
def train_datamodule(
    datamodule: DataModule[AsConverter, _InputT],
    dataset: Iterable[_InputT],
) -> DataModule[AsConverter, _InputT]:
    with datamodule.train(), progress(dataset, desc="Training datamodule") as dataset:
        for example in dataset:
            datamodule(example)
    return datamodule


@step("ml::train_datamodule_with_instances", format=DataModuleAndInstancesFormat())
def train_datamodule_with_instances(
    datamodule: DataModule[AsConverter, _InputT, _InstanceT],
    dataset: Iterable[_InputT],
) -> DataModuleAndInstances[_InputT, _InstanceT]:
    def generate_instances() -> Iterator[_InstanceT]:
        nonlocal datamodule, dataset

        with datamodule.train(), progress(dataset, desc="Training datamodule") as dataset:
            for example in dataset:
                instance = datamodule(example)
                assert instance is not None
                yield instance

    return DataModuleAndInstances(datamodule=datamodule, instances=generate_instances())


@step("ml::generate_instances", format="dataset")
@step("ml::generate_instances_without_caching", cacheable=False)
def generate_instances(
    datamodule: DataModule[AsConverter, _InputT, _InstanceT],
    dataset: Iterable[_InputT],
) -> Dataset[_InstanceT]:
    def generator() -> Iterator[_InstanceT]:
        nonlocal datamodule, dataset
        with progress(dataset, desc="Generating instances") as dataset:
            for example in dataset:
                instance = datamodule(example)
                assert instance is not None
                yield instance

    return Dataset.from_iterable(generator())
