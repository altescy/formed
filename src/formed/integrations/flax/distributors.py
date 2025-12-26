"""Distributed computing abstractions for Flax models.

This module provides abstractions for distributed training across multiple devices,
supporting both single-device and data-parallel training strategies.

Key Components:
    - `BaseDistributor`: Abstract interface for device distribution strategies
    - `SingleDeviceDistributor`: No-op distributor for single-device training
    - `DataParallelDistributor`: Data-parallel training using JAX pmap

Features:
    - Transparent device sharding and replication
    - JIT compilation with device-specific optimizations
    - Reduction operations (mean, sum) across devices
    - Compatible with FlaxTrainer

Examples:
    >>> from formed.integrations.flax import DataParallelDistributor
    >>> import jax
    >>>
    >>> # Create data-parallel distributor for all available devices
    >>> distributor = DataParallelDistributor(axis_name="batch")
    >>>
    >>> # Shard batch across devices
    >>> sharded_batch = distributor.shard(batch)
    >>>
    >>> # Map function across devices
    >>> train_step = distributor.map(training_function)
    >>> outputs = train_step(sharded_batch, state)

"""

import abc
from collections.abc import Callable, Sequence
from typing import Generic, Literal, TypeAlias, TypeVar, cast

import flax.jax_utils
import jax
from colt import Registrable
from flax import nnx

from .types import ModelInputT

_T = TypeVar("_T")
_ArrayT = TypeVar("_ArrayT", bound=jax.Array)
_CallableT = TypeVar("_CallableT", bound=Callable)
_ReduceOp: TypeAlias = Literal["mean", "sum"]


class BaseDistributor(Registrable, abc.ABC, Generic[ModelInputT]):
    """Abstract base class for device distribution strategies.

    BaseDistributor defines the interface for distributing computations
    across devices in a JAX/Flax training pipeline. It handles data
    sharding, replication, and reduction operations.

    Type Parameters:
        ModelInputT: Type of model input data.

    """

    def shard(self, inputs: ModelInputT) -> ModelInputT:
        """Shard inputs across devices.

        Args:
            inputs: Input data to shard.

        Returns:
            Sharded input data with an additional device dimension.

        """
        return inputs

    def replicate(self, inputs: _T) -> _T:
        """Replicate data across all devices.

        Args:
            inputs: Data to replicate.

        Returns:
            Replicated data with device dimension.

        """
        return inputs

    def unreplicate(self, inputs: _T) -> _T:
        """Extract data from the first device, removing device dimension.

        Args:
            inputs: Replicated data with device dimension.

        Returns:
            Data from first device without device dimension.

        """
        return inputs

    @abc.abstractmethod
    def map(self, fn: _CallableT, static_argnums: Sequence[int] = ()) -> _CallableT:
        """Map a function across devices with JIT compilation.

        Args:
            fn: Function to map across devices.
            static_argnums: Indices of static arguments.

        Returns:
            Mapped and compiled function.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def reduce(self, array: _ArrayT, op: _ReduceOp = "mean") -> _ArrayT:
        """Reduce an array across devices.

        Args:
            array: Array to reduce.
            op: Reduction operation (`"mean"` or `"sum"`).

        Returns:
            Reduced array.

        """
        raise NotImplementedError


@BaseDistributor.register("single")
class SingleDeviceDistributor(BaseDistributor[ModelInputT]):
    """Distributor for single-device training.

    This distributor applies JIT compilation without any device distribution.
    All shard, replicate, and unreplicate operations are no-ops.

    Examples:
        >>> distributor = SingleDeviceDistributor()
        >>> train_step = distributor.map(my_train_function)
        >>> output = train_step(batch, state, trainer)

    """

    def map(self, fn: _CallableT, static_argnums: Sequence[int] = ()) -> _CallableT:
        """Apply JIT compilation to a function.

        Args:
            fn: Function to compile.
            static_argnums: Indices of static arguments.

        Returns:
            JIT-compiled function.

        """
        return cast(_CallableT, nnx.jit(fn, static_argnums=static_argnums))

    def reduce(self, array: _ArrayT, op: _ReduceOp = "mean") -> _ArrayT:
        """Return array unchanged (no reduction needed for single device).

        Args:
            array: Input array.
            op: Reduction operation (ignored).

        Returns:
            Input array unchanged.

        """
        return array


@BaseDistributor.register("data_parallel")
class DataParallelDistributor(BaseDistributor[ModelInputT]):
    """Distributor for data-parallel training across multiple devices.

    This distributor uses JAX's pmap to execute the same computation on
    different data shards across multiple devices. Data is automatically
    sharded along the batch dimension.

    Args:
        axis_name: Name for the device axis (used in reduction operations).
        num_devices: Number of devices to use. Defaults to all local devices.

    Examples:
        >>> # Train on 4 GPUs with data parallelism
        >>> distributor = DataParallelDistributor(
        ...     axis_name="batch",
        ...     num_devices=4
        ... )
        >>>
        >>> # Shard batch of size 32 into 4 shards of size 8
        >>> sharded = distributor.shard(batch)
        >>> assert sharded.shape == (4, 8, ...)
        >>>
        >>> # Map training step across devices
        >>> train_step = distributor.map(my_train_step)

    Note:
        Batch size must be divisible by num_devices for proper sharding.

    """

    def __init__(
        self,
        axis_name: str = "batch",
        num_devices: int | None = None,
    ) -> None:
        self._axis_name = axis_name
        self._num_devices = num_devices or jax.local_device_count()

    def shard(self, inputs: ModelInputT) -> ModelInputT:
        """Shard inputs along the batch dimension across devices.

        Args:
            inputs: Input data with batch dimension.

        Returns:
            Sharded inputs with shape (num_devices, batch_per_device, ...).

        """
        return jax.tree_util.tree_map(lambda x: x.reshape((self._num_devices, -1) + x.shape[1:]), inputs)

    def replicate(self, inputs: _T) -> _T:
        """Replicate data across all devices.

        Args:
            inputs: Data to replicate.

        Returns:
            Replicated data with device dimension.

        """
        return flax.jax_utils.replicate(inputs)

    def unreplicate(self, inputs: _T) -> _T:
        """Extract data from the first device.

        Args:
            inputs: Replicated data.

        Returns:
            Data from first device.

        """
        return flax.jax_utils.unreplicate(inputs)

    def map(self, fn: _CallableT, static_argnums: Sequence[int] = ()) -> _CallableT:
        """Map function across devices using pmap.

        Args:
            fn: Function to parallelize.
            static_argnums: Indices of static arguments to broadcast.

        Returns:
            Parallelized function using pmap.

        """
        return nnx.pmap(fn, axis_name=self._axis_name, static_broadcasted_argnums=static_argnums)

    def reduce(self, array: _ArrayT, op: _ReduceOp = "mean") -> _ArrayT:
        """Reduce array across devices.

        Args:
            array: Array to reduce across device dimension.
            op: Reduction operation - `"sum"` or `"mean"`.

        Returns:
            Reduced array.

        Raises:
            ValueError: If unsupported reduction operation is specified.

        """
        if op == "sum":
            return jax.lax.psum(array, axis_name=self._axis_name)
        elif op == "mean":
            return jax.lax.pmean(array, axis_name=self._axis_name)
        raise ValueError(f"Unsupported reduce operation: {op}")
