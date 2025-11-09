import abc
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar, cast

import flax.jax_utils
import jax
from colt import Registrable
from flax import nnx

from .types import ModelInputT

_T = TypeVar("_T")
_ArrayT = TypeVar("_ArrayT", bound=jax.Array)
_CallableT = TypeVar("_CallableT", bound=Callable)


class BaseDistributor(Registrable, abc.ABC, Generic[ModelInputT]):
    def shard(self, inputs: ModelInputT) -> ModelInputT:
        return inputs

    def replicate(self, inputs: _T) -> _T:
        return inputs

    def unreplicate(self, inputs: _T) -> _T:
        return inputs

    @abc.abstractmethod
    def map(self, fn: _CallableT, static_argnums: Sequence[int] = ()) -> _CallableT:
        raise NotImplementedError

    @abc.abstractmethod
    def reduce(self, array: _ArrayT) -> _ArrayT:
        raise NotImplementedError


@BaseDistributor.register("single")
class SingleDeviceDistributor(BaseDistributor[ModelInputT]):
    def map(self, fn: _CallableT, static_argnums: Sequence[int] = ()) -> _CallableT:
        return cast(_CallableT, nnx.jit(fn, static_argnums=static_argnums))

    def reduce(self, array: _ArrayT) -> _ArrayT:
        return array


@BaseDistributor.register("data_parallel")
class DataParallelDistributor(BaseDistributor[ModelInputT]):
    def __init__(self, axis_name: str = "batch") -> None:
        self._axis_name = axis_name

    def shard(self, inputs: ModelInputT) -> ModelInputT:
        n = jax.local_device_count()
        return jax.tree_util.tree_map(lambda x: x.reshape((n, -1) + x.shape[1:]), inputs)

    def replicate(self, inputs: _T) -> _T:
        return flax.jax_utils.replicate(inputs)

    def unreplicate(self, inputs: _T) -> _T:
        return flax.jax_utils.unreplicate(inputs)

    def map(self, fn: _CallableT, static_argnums: Sequence[int] = ()) -> _CallableT:
        return nnx.pmap(fn, axis_name=self._axis_name, static_broadcasted_argnums=static_argnums)

    def reduce(self, array: _ArrayT) -> _ArrayT:
        return jax.lax.pmean(array, axis_name=self._axis_name)
