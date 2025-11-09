import abc
from collections.abc import Callable, Sequence
from typing import Generic, Literal, Optional, TypeVar, cast

import flax.jax_utils
import jax
from colt import Registrable
from flax import nnx
from typing_extensions import TypeAlias

from .types import ModelInputT

_T = TypeVar("_T")
_ArrayT = TypeVar("_ArrayT", bound=jax.Array)
_CallableT = TypeVar("_CallableT", bound=Callable)
_ReduceOp: TypeAlias = Literal["mean", "sum"]


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
    def reduce(self, array: _ArrayT, op: _ReduceOp = "mean") -> _ArrayT:
        raise NotImplementedError


@BaseDistributor.register("single")
class SingleDeviceDistributor(BaseDistributor[ModelInputT]):
    def map(self, fn: _CallableT, static_argnums: Sequence[int] = ()) -> _CallableT:
        return cast(_CallableT, nnx.jit(fn, static_argnums=static_argnums))

    def reduce(self, array: _ArrayT, op: _ReduceOp = "mean") -> _ArrayT:
        return array


@BaseDistributor.register("data_parallel")
class DataParallelDistributor(BaseDistributor[ModelInputT]):
    def __init__(
        self,
        axis_name: str = "batch",
        num_devices: Optional[int] = None,
    ) -> None:
        self._axis_name = axis_name
        self._num_devices = num_devices or jax.local_device_count()

    def shard(self, inputs: ModelInputT) -> ModelInputT:
        return jax.tree_util.tree_map(lambda x: x.reshape((self._num_devices, -1) + x.shape[1:]), inputs)

    def replicate(self, inputs: _T) -> _T:
        return flax.jax_utils.replicate(inputs)

    def unreplicate(self, inputs: _T) -> _T:
        return flax.jax_utils.unreplicate(inputs)

    def map(self, fn: _CallableT, static_argnums: Sequence[int] = ()) -> _CallableT:
        return nnx.pmap(fn, axis_name=self._axis_name, static_broadcasted_argnums=static_argnums)

    def reduce(self, array: _ArrayT, op: _ReduceOp = "mean") -> _ArrayT:
        if op == "sum":
            return jax.lax.psum(array, axis_name=self._axis_name)
        elif op == "mean":
            return jax.lax.pmean(array, axis_name=self._axis_name)
        raise ValueError(f"Unsupported reduce operation: {op}")
