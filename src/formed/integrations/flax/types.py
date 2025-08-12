import dataclasses
from collections.abc import Iterator, Mapping
from typing import ClassVar, Optional, Protocol, TypeVar, runtime_checkable

import jax
import optax

from formed.integrations.ml import Field


@runtime_checkable
class IOptimizer(Protocol):
    init: optax.TransformInitFn
    update: optax.TransformUpdateFn


@runtime_checkable
class IModelInput(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field]]


@runtime_checkable
class IModelOutput(Protocol):
    loss: Optional[jax.Array]
    metrics: Optional[Mapping[str, jax.Array]]


DataT = TypeVar("DataT", bound=Mapping[str, Field])
ModelInputT = TypeVar("ModelInputT", bound=IModelInput)
ModelParamsT = TypeVar("ModelParamsT")
ModelOutputT = TypeVar("ModelOutputT", bound=IModelOutput)
ModelInputT_co = TypeVar("ModelInputT_co", covariant=True)


class IBatchIterator(Protocol[ModelInputT_co]):
    def __iter__(self) -> Iterator[ModelInputT_co]: ...

    def __len__(self) -> int: ...
