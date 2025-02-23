from collections.abc import Sequence
from typing import Optional, TypeVar

from formed.workflow import step

from .datamodule import DataModule

T = TypeVar("T")


@step("formedml::build_datamodule", format="pickle")
def build_datamodule(
    dataset: Sequence[T],
    datamodule: Optional[DataModule[T]] = None,
) -> DataModule[T]:
    datamodule = datamodule or DataModule()
    datamodule.build(dataset)
    return datamodule
