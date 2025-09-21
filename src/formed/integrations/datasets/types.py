from collections.abc import Mapping
from typing import TypeVar, Union

import datasets

Dataset = Union[datasets.Dataset, datasets.DatasetDict]
DatasetT = TypeVar("DatasetT", bound=Dataset)
DatasetOrMappingT = TypeVar("DatasetOrMappingT", bound=Union[Dataset, Mapping[str, Dataset]])
