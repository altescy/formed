from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, TypeVar

import jax
from flax import nnx

from formed.integrations.flax.utils import sequence_distribute, sequence_undistribute

T = TypeVar("T", jax.Array, Mapping)


class SequenceDistributed(nnx.Module, Generic[T]):
    def __init__(
        self,
        module: Callable[..., T],
        ignore: Sequence[str] = (),
    ) -> None:
        self.module: Callable[..., T] = module
        self.ignore: Sequence[str] = ignore

    def __call__(self, **kwargs: Any) -> T:
        inputs, shape = sequence_distribute(kwargs, ignore=self.ignore)
        result = self.module(**inputs)
        return sequence_undistribute(result, shape)
