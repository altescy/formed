"""DataLoader utilities for PyTorch training.

This module provides convenient wrappers for creating PyTorch DataLoaders
that work seamlessly with the formed training framework.

Examples:
    >>> from formed.integrations.torch import DataLoader
    >>>
    >>> # Create a simple dataloader
    >>> train_loader = DataLoader(
    ...     batch_size=32,
    ...     shuffle=True,
    ...     collate_fn=my_collate_fn
    ... )
    >>>
    >>> # Use with trainer
    >>> trainer = TorchTrainer(
    ...     train_dataloader=train_loader,
    ...     ...
    ... )

"""

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch.utils.data

from .types import IBatchIterator, ModelInputT

ItemT = TypeVar("ItemT")


class DataLoader:
    """Simple DataLoader wrapper for PyTorch training.

    This class wraps PyTorch's DataLoader with a simpler interface
    that works with the formed training framework.

    Args:
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at every epoch.
        collate_fn: Function to collate samples into batches.
        num_workers: Number of subprocesses for data loading.
        drop_last: Whether to drop the last incomplete batch.
        pin_memory: If True, tensors are copied to CUDA pinned memory.
        **kwargs: Additional arguments passed to torch.utils.data.DataLoader.

    Examples:
        >>> def collate_fn(batch):
        ...     # Convert list of samples to batch tensors
        ...     return {"features": torch.stack([x["features"] for x in batch])}
        >>>
        >>> loader = DataLoader(
        ...     batch_size=32,
        ...     shuffle=True,
        ...     collate_fn=collate_fn
        ... )

    """

    def __init__(
        self,
        batch_size: int,
        shuffle: bool = False,
        collate_fn: Callable[[list[ItemT]], ModelInputT] | None = None,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def __call__(self, data: Sequence[ItemT]) -> IBatchIterator[ModelInputT]:
        """Create a PyTorch DataLoader for the given dataset.

        Args:
            data: Sequence of data samples.

        Returns:
            PyTorch DataLoader that yields batches.

        """
        return torch.utils.data.DataLoader(
            data,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            **self.kwargs,
        )
