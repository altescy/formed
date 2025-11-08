from collections.abc import Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

import optax
from typing_extensions import TypeVar


@runtime_checkable
class IOptimizer(Protocol):
    init: optax.TransformInitFn
    update: optax.TransformUpdateFn


ItemT = TypeVar("ItemT", default=Any)
ItemT_contra = TypeVar("ItemT_contra", contravariant=True, default=Any)
ModelInputT = TypeVar("ModelInputT", default=Any)
ModelInputT_co = TypeVar("ModelInputT_co", covariant=True, default=Any)
ModelOutputT = TypeVar("ModelOutputT", default=Any)
ModelParamsT = TypeVar("ModelParamsT", default=None)
OptimizerT = TypeVar("OptimizerT", bound=IOptimizer)


class IBatchIterator(Protocol[ModelInputT_co]):
    def __iter__(self) -> Iterator[ModelInputT_co]: ...

    def __len__(self) -> int: ...


class IDataLoader(Protocol[ItemT_contra, ModelInputT_co]):
    def __call__(self, data: Sequence[ItemT_contra]) -> IBatchIterator[ModelInputT_co]: ...
