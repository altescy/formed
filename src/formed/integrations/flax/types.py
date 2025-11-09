from collections.abc import Iterator, Sequence
from typing import Any, Optional, Protocol, Union, runtime_checkable

import jax
import numpy
import optax
from typing_extensions import TypeAlias, TypeVar


@runtime_checkable
class IOptimizer(Protocol):
    init: optax.TransformInitFn
    update: optax.TransformUpdateFn


ArrayCompatible: TypeAlias = Union[numpy.ndarray, jax.Array]
ArrayCompatibleT = TypeVar("ArrayCompatibleT", bound=ArrayCompatible)
ItemT = TypeVar("ItemT", default=Any)
ItemT_contra = TypeVar("ItemT_contra", contravariant=True, default=Any)
ModelInputT = TypeVar("ModelInputT", default=Any)
ModelInputT_co = TypeVar("ModelInputT_co", covariant=True, default=Any)
ModelInputT_contra = TypeVar("ModelInputT_contra", contravariant=True, default=Any)
ModelOutputT = TypeVar("ModelOutputT", default=Any)
ModelOutputT_contra = TypeVar("ModelOutputT_contra", contravariant=True, default=Any)
ModelParamsT = TypeVar("ModelParamsT", default=None)
OptimizerT = TypeVar("OptimizerT", bound=IOptimizer)


class IBatchIterator(Protocol[ModelInputT_co]):
    def __iter__(self) -> Iterator[ModelInputT_co]: ...
    def __len__(self) -> int: ...


class IDataLoader(Protocol[ItemT_contra, ModelInputT_co]):
    def __call__(self, data: Sequence[ItemT_contra]) -> IBatchIterator[ModelInputT_co]: ...


class IEvaluator(Protocol[ModelInputT_contra, ModelOutputT_contra]):
    def update(self, inputs: ModelInputT_contra, output: ModelOutputT_contra, /) -> None: ...
    def compute(self) -> dict[str, float]: ...
    def reset(self) -> None: ...


class IIDSequenceBatch(Protocol[ArrayCompatibleT]):
    ids: ArrayCompatibleT
    mask: ArrayCompatibleT

    def __len__(self) -> int: ...


IDSequenceBatchT = TypeVar("IDSequenceBatchT", bound=IIDSequenceBatch)


SurfaceBatchT = TypeVar("SurfaceBatchT", bound=IIDSequenceBatch, default=Any)
PostagBatchT = TypeVar("PostagBatchT", bound=Optional[IIDSequenceBatch], default=Any)
CharacterBatchT = TypeVar("CharacterBatchT", bound=Optional[IIDSequenceBatch], default=Any)


class IAnalyzedTextBatch(Protocol[SurfaceBatchT, PostagBatchT, CharacterBatchT]):
    surfaces: SurfaceBatchT
    postags: PostagBatchT
    characters: CharacterBatchT

    def __len__(self) -> int: ...


AnalyzedTextBatchT = TypeVar("AnalyzedTextBatchT", bound=IAnalyzedTextBatch, default=Any)
