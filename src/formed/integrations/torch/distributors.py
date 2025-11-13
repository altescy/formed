"""Distributed computing abstractions for PyTorch models.

This module provides abstractions for distributed training across multiple devices,
supporting both single-device and data-parallel training strategies.

Key Components:
    - BaseDistributor: Abstract interface for device distribution strategies
    - SingleDeviceDistributor: No-op distributor for single-device training
    - DataParallelDistributor: Data-parallel training using torch.nn.DataParallel

Features:
    - Transparent device sharding and replication
    - Reduction operations (mean, sum) across devices
    - Compatible with TorchTrainer

Example:
    >>> from formed.integrations.torch import DataParallelDistributor
    >>> import torch
    >>>
    >>> # Create data-parallel distributor for all available GPUs
    >>> distributor = DataParallelDistributor()
    >>>
    >>> # Shard batch across devices
    >>> sharded_batch = distributor.shard(batch)

"""

import abc
from collections.abc import Sequence
from typing import Generic, Literal, TypeAlias, TypeVar, cast

import torch
import torch.nn as nn
from colt import Registrable

from .types import ModelInputT

_T = TypeVar("_T")
_TensorT = TypeVar("_TensorT", bound=torch.Tensor)
_ReduceOp: TypeAlias = Literal["mean", "sum"]


class BaseDistributor(Registrable, abc.ABC, Generic[ModelInputT]):
    """Abstract base class for device distribution strategies.

    BaseDistributor defines the interface for distributing computations
    across devices in a PyTorch training pipeline. It provides a unified
    API for single-device, data-parallel, and distributed data-parallel training.

    Type Parameters:
        ModelInputT: Type of model input data.

    Key Methods:
        - device: Primary device for computation
        - is_main_process: Whether this is the main process (for logging, saving, etc.)
        - wrap_model: Wrap model for distributed training
        - prepare_data_loader: Prepare data loader with appropriate sampler
        - reduce: Reduce tensor across devices/processes
        - barrier: Synchronize all processes
        - all_gather: Gather tensors from all processes

    """

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Primary device for computation."""
        raise NotImplementedError

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process.

        The main process is responsible for:
        - Logging to console
        - Saving models and checkpoints
        - Writing metrics to file

        Returns:
            True if this is the main process (rank 0), False otherwise.

        """
        return True

    @property
    def world_size(self) -> int:
        """Total number of processes/devices.

        Returns:
            Number of processes in distributed training, or 1 for single device.

        """
        return 1

    @property
    def rank(self) -> int:
        """Global rank of this process.

        Returns:
            Rank of this process (0 for main process).

        """
        return 0

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training.

        Args:
            model: Model to wrap.

        Returns:
            Wrapped model (DataParallel, DDP, or unchanged).

        """
        return model

    def prepare_data_loader(
        self,
        dataset: Sequence,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """Prepare data loader with appropriate sampler for this distributor.

        For single device: uses default sampler
        For DataParallel: uses default sampler (data split happens in forward)
        For DDP: uses DistributedSampler to split data across processes

        Args:
            dataset: Dataset to load.
            batch_size: Batch size per device/process.
            shuffle: Whether to shuffle data.
            num_workers: Number of worker processes.
            drop_last: Whether to drop last incomplete batch.
            **kwargs: Additional arguments for DataLoader.

        Returns:
            Configured DataLoader.

        """
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset,  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            **kwargs,
        )

    @abc.abstractmethod
    def reduce(self, tensor: _TensorT, op: _ReduceOp = "mean") -> _TensorT:
        """Reduce a tensor across devices/processes.

        Args:
            tensor: Tensor to reduce.
            op: Reduction operation ("mean" or "sum").

        Returns:
            Reduced tensor.

        """
        raise NotImplementedError

    def barrier(self) -> None:
        """Synchronize all processes.

        This is a no-op for single device and DataParallel.
        For DDP, it blocks until all processes reach this point.

        """
        pass

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Gather tensors from all processes.

        Args:
            tensor: Tensor to gather.

        Returns:
            List of tensors from all processes.
            For single device/DataParallel, returns [tensor].

        """
        return [tensor]

    def cleanup(self) -> None:
        """Cleanup resources (e.g., distributed process group).

        This is a no-op for single device and DataParallel.
        For DDP, destroys the process group.

        """
        pass


@BaseDistributor.register("single")
class SingleDeviceDistributor(BaseDistributor[ModelInputT]):
    """Distributor for single-device training.

    This distributor operates on a single device without any distribution.
    All shard, replicate, and unreplicate operations are no-ops.

    Args:
        device: Device to use (default: "cuda" if available, else "cpu").

    Example:
        >>> distributor = SingleDeviceDistributor(device="cuda:0")
        >>> model = model.to(distributor.device)

    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def reduce(self, tensor: _TensorT, op: _ReduceOp = "mean") -> _TensorT:
        """Return tensor unchanged (no reduction needed for single device).

        Args:
            tensor: Input tensor.
            op: Reduction operation (ignored).

        Returns:
            Input tensor unchanged.

        """
        return tensor


@BaseDistributor.register("data_parallel")
class DataParallelDistributor(BaseDistributor[ModelInputT]):
    """Distributor for data-parallel training across multiple GPUs.

    This distributor uses torch.nn.DataParallel to execute the same computation
    on different data shards across multiple GPUs. Data is automatically
    sharded along the batch dimension.

    Args:
        device_ids: List of GPU device IDs to use. Defaults to all available GPUs.
        output_device: Device for outputs. Defaults to device_ids[0].

    Example:
        >>> # Train on GPUs 0 and 1 with data parallelism
        >>> distributor = DataParallelDistributor(device_ids=[0, 1])
        >>>
        >>> # Wrap model for data parallel training
        >>> model = distributor.wrap_model(model)

    Note:
        Batch size must be divisible by the number of devices for proper sharding.

    """

    def __init__(
        self,
        device_ids: list[int] | None = None,
        output_device: int | None = None,
    ) -> None:
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if not device_ids:
            raise ValueError("No GPU devices available for DataParallelDistributor")

        self._device_ids = device_ids
        self._output_device = output_device if output_device is not None else device_ids[0]
        self._device = torch.device(f"cuda:{self._output_device}")

    @property
    def device(self) -> torch.device:
        return self._device

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with DataParallel.

        Args:
            model: Model to wrap.

        Returns:
            DataParallel wrapped model.

        """
        return cast(nn.Module, nn.DataParallel(model, device_ids=self._device_ids, output_device=self._output_device))

    def reduce(self, tensor: _TensorT, op: _ReduceOp = "mean") -> _TensorT:
        """Reduce tensor across devices.

        Args:
            tensor: Tensor to reduce across device dimension.
            op: Reduction operation - "sum" or "mean".

        Returns:
            Reduced tensor.

        Raises:
            ValueError: If unsupported reduction operation is specified.

        """
        if op == "sum":
            return cast(_TensorT, tensor.sum())
        elif op == "mean":
            return cast(_TensorT, tensor.mean())
        raise ValueError(f"Unsupported reduce operation: {op}")


@BaseDistributor.register("distributed_data_parallel")
class DistributedDataParallelDistributor(BaseDistributor[ModelInputT]):
    """Distributor for distributed data-parallel training using DDP.

    This distributor uses torch.nn.parallel.DistributedDataParallel to execute
    training across multiple processes and devices. This is more efficient than
    DataParallel for multi-GPU training as it uses one process per GPU.

    Args:
        backend: Backend to use for distributed training ("nccl", "gloo", "mpi").
            Defaults to "nccl" for GPU and "gloo" for CPU.
        init_method: URL specifying how to initialize the process group.
            Defaults to "env://" which uses environment variables.
        world_size: Total number of processes. If None, reads from environment.
        rank: Rank of this process. If None, reads from environment.
        local_rank: Local rank on this machine. If None, uses rank.
        find_unused_parameters: Whether to find unused parameters. Default False.
        broadcast_buffers: Whether to broadcast buffers. Default True.
        bucket_cap_mb: Bucket size in MB for gradient allreduce. Default 25.

    Environment Variables:
        - RANK: Global rank of the process
        - LOCAL_RANK: Local rank on the machine
        - WORLD_SIZE: Total number of processes
        - MASTER_ADDR: Address of the master node
        - MASTER_PORT: Port of the master node

    Example:
        >>> # On each process, initialize the distributor
        >>> distributor = DistributedDataParallelDistributor(
        ...     backend="nccl",
        ...     init_method="env://"
        ... )
        >>>
        >>> # Wrap model with DDP
        >>> model = distributor.wrap_model(model)
        >>>
        >>> # Train as usual - gradients are automatically synchronized

    Note:
        - Requires launching multiple processes (e.g., using torch.distributed.launch)
        - Each process should initialize its own distributor
        - Batch size should be per-process batch size

    """

    def __init__(
        self,
        backend: str | None = None,
        init_method: str = "env://",
        world_size: int | None = None,
        rank: int | None = None,
        local_rank: int | None = None,
        find_unused_parameters: bool = False,
        broadcast_buffers: bool = True,
        bucket_cap_mb: int = 25,
    ) -> None:
        import os

        import torch.distributed as dist

        # Determine backend
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        # Get rank and world_size from environment if not provided
        if rank is None:
            rank = int(os.environ.get("RANK", 0))
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", rank))

        self._backend = backend
        self._init_method = init_method
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._find_unused_parameters = find_unused_parameters
        self._broadcast_buffers = broadcast_buffers
        self._bucket_cap_mb = bucket_cap_mb

        # Initialize process group if not already initialized
        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend=backend,
                    init_method=init_method,
                    world_size=world_size,
                    rank=rank,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize distributed process group. "
                    f"Backend: {backend}, init_method: {init_method}, "
                    f"world_size: {world_size}, rank: {rank}. "
                    f"Error: {e}. "
                    "Please ensure all processes are launched correctly using torchrun "
                    "and required environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) are set."
                ) from e

        # Set device based on local rank
        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self._device)
        else:
            self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process (rank 0)."""
        return self._rank == 0

    @property
    def rank(self) -> int:
        """Global rank of this process."""
        return self._rank

    @property
    def local_rank(self) -> int:
        """Local rank on this machine."""
        return self._local_rank

    @property
    def world_size(self) -> int:
        """Total number of processes."""
        return self._world_size

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with DistributedDataParallel.

        Args:
            model: Model to wrap.

        Returns:
            DDP wrapped model.

        """
        return cast(
            nn.Module,
            nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self._local_rank] if torch.cuda.is_available() else None,
                output_device=self._local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=self._find_unused_parameters,
                broadcast_buffers=self._broadcast_buffers,
                bucket_cap_mb=self._bucket_cap_mb,
            ),
        )

    def prepare_data_loader(
        self,
        dataset: Sequence,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """Prepare data loader with DistributedSampler for DDP.

        Args:
            dataset: Dataset to load.
            batch_size: Batch size per process.
            shuffle: Whether to shuffle data.
            num_workers: Number of worker processes.
            drop_last: Whether to drop last incomplete batch.
            **kwargs: Additional arguments for DataLoader.

        Returns:
            DataLoader with DistributedSampler.

        """
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,  # type: ignore[arg-type]
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return DataLoader(
            dataset,  # type: ignore[arg-type]
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )

    def reduce(self, tensor: _TensorT, op: _ReduceOp = "mean") -> _TensorT:
        """Reduce tensor across all processes.

        Args:
            tensor: Tensor to reduce.
            op: Reduction operation - "sum" or "mean".

        Returns:
            Reduced tensor.

        Raises:
            ValueError: If unsupported reduction operation is specified.

        """
        import torch.distributed as dist

        if op == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor
        elif op == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return cast(_TensorT, tensor / self._world_size)
        raise ValueError(f"Unsupported reduce operation: {op}")

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Gather tensors from all processes.

        Args:
            tensor: Tensor to gather.

        Returns:
            List of tensors from all processes.

        """
        import torch.distributed as dist

        gathered = [torch.zeros_like(tensor) for _ in range(self._world_size)]
        dist.all_gather(gathered, tensor)
        return gathered

    def barrier(self) -> None:
        """Synchronize all processes.

        This creates a barrier that blocks until all processes reach this point.

        """
        import torch.distributed as dist

        dist.barrier()

    def cleanup(self) -> None:
        """Cleanup distributed process group.

        This should be called at the end of training.

        """
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
