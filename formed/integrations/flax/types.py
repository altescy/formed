from collections.abc import Iterator
from typing import Mapping, Optional, Protocol, TypeVar, runtime_checkable

import jax
import optax


@runtime_checkable
class IOptimizer(Protocol):
    init: optax.TransformInitFn
    update: optax.TransformUpdateFn


@runtime_checkable
class IModelOutput(Protocol):
    loss: Optional[jax.Array]
    metrics: Optional[Mapping[str, jax.Array]]


DataT = TypeVar("DataT")
ModelInputT = TypeVar("ModelInputT")
ModelParamsT = TypeVar("ModelParamsT")
ModelOutputT = TypeVar("ModelOutputT", bound=IModelOutput)
ModelInputT_co = TypeVar("ModelInputT_co", covariant=True)


class IBatchIterator(Protocol[ModelInputT_co]):
    def __iter__(self) -> Iterator[ModelInputT_co]: ...

    def __len__(self) -> int: ...
