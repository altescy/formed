from pathlib import Path
from typing import Any, Generic, Iterable, Iterator, NamedTuple

from typing_extensions import TypeVar

from formed.common.rich import progress
from formed.workflow import step
from formed.workflow.format import Format, JsonFormat, PickleFormat

from .transforms import DataModule
from .types import AsConverter, AsInstance

InputT = TypeVar("InputT", default=Any)
InstanceT = TypeVar("InstanceT", bound=DataModule[AsInstance], default=Any)
DataModuleT = TypeVar("DataModuleT", bound=DataModule[AsConverter], default=Any)


class DataModuleAndInstances(NamedTuple, Generic[InputT, InstanceT]):
    datamodule: DataModule[AsConverter, InputT, InstanceT]
    instances: Iterable[InstanceT]


@Format.register("ml::datamodule_and_dataset")
class DataModuleAndInstancesFormat(
    Format[DataModuleAndInstances[InputT, InstanceT]],
    Generic[InputT, InstanceT],
):
    _INSTANCES_FORMAT = PickleFormat[Iterable[InstanceT]]()
    _DATAMODULE_FORMAT = JsonFormat[DataModule[AsConverter, InputT]]()

    def write(
        self,
        artifact: DataModuleAndInstances[InputT, InstanceT],
        directory: Path,
    ) -> None:
        instances_path = directory / "instances"
        datamodule_path = directory / "datamodule"

        instances_path.mkdir(parents=True, exist_ok=True)
        datamodule_path.mkdir(parents=True, exist_ok=True)

        self._INSTANCES_FORMAT.write(artifact.instances, instances_path)
        self._DATAMODULE_FORMAT.write(artifact.datamodule, datamodule_path)

    def read(self, directory: Path) -> DataModuleAndInstances[InputT, InstanceT]:
        instances_path = directory / "instances"
        datamodule_path = directory / "datamodule"

        instances = self._INSTANCES_FORMAT.read(instances_path)
        datamodule = self._DATAMODULE_FORMAT.read(datamodule_path)

        return DataModuleAndInstances(datamodule=datamodule, instances=instances)


@step("ml::train_datamodule", format="json")
def train_datamodule(
    datamodule: DataModule[AsConverter, InputT],
    dataset: Iterable[InputT],
) -> DataModule[AsConverter, InputT]:
    with datamodule.train(), progress(dataset, desc="Training datamodule") as dataset:
        for example in dataset:
            datamodule(example)
    return datamodule


@step("ml::train_datamodule_with_instances", format=DataModuleAndInstancesFormat())
def train_datamodule_with_instances(
    datamodule: DataModule[AsConverter, InputT, InstanceT],
    dataset: Iterable[InputT],
) -> DataModuleAndInstances[InputT, InstanceT]:
    def generate_instances() -> Iterator[InstanceT]:
        nonlocal datamodule, dataset

        with datamodule.train(), progress(dataset, desc="Training datamodule") as dataset:
            for example in dataset:
                instance = datamodule(example)
                assert instance is not None
                yield instance

    return DataModuleAndInstances(datamodule=datamodule, instances=generate_instances())
