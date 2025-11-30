import abc
import math
import random
from collections.abc import Callable, Iterator, Sequence
from typing import Generic, TypeVar

from colt import Registrable

from formed.common.iterutils import BufferedIterator, SizedIterator

_InputT = TypeVar("_InputT")
_BatchT = TypeVar("_BatchT")


class BaseBatchSampler(Registrable, abc.ABC):
    """Abstract base class for batch samplers.

    Batch samplers generate sequences of indices for batching data.
    """

    @abc.abstractmethod
    def __call__(self, data: Sequence) -> SizedIterator[Sequence[int]]:
        """Generate batch indices from a dataset.

        Args:
            data: The dataset to sample from.

        Returns:
            A sized iterator of batch indices.
        """
        raise NotImplementedError


@BaseBatchSampler.register("basic")
class BasicBatchSampler(BaseBatchSampler):
    """Basic batch sampler that supports shuffling and dropping incomplete batches.

    Args:
        batch_size: Number of samples per batch. Defaults to 1.
        shuffle: Whether to shuffle the data before batching. Defaults to False.
        drop_last: Whether to drop the last incomplete batch. Defaults to False.
        seed: Random seed for shuffling. Defaults to 0.

    Examples:
        >>> sampler = BasicBatchSampler(batch_size=32, shuffle=True)
        >>> dataset = list(range(100))
        >>> batches = list(sampler(dataset))
        >>> len(batches)
        4
    """

    def __init__(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._rng = random.Random(seed)

    def __call__(self, data: Sequence) -> SizedIterator[Sequence[int]]:
        """Generate batch indices from a dataset.

        Args:
            data: The dataset to sample from.

        Returns:
            A sized iterator of batch indices. Each batch is a sequence of
            indices into the original dataset.
        """
        indices = list(range(len(data)))
        if self._shuffle:
            self._rng.shuffle(indices)
        if self._drop_last:
            indices = indices[: len(indices) - len(indices) % self._batch_size]

        size = math.ceil(len(indices) / self._batch_size)

        def iterator() -> Iterator[Sequence[int]]:
            for i in range(0, len(indices), self._batch_size):
                yield indices[i : i + self._batch_size]

        return SizedIterator(iterator(), size)


class BatchIterator(Generic[_InputT, _BatchT]):
    """Iterator that generates batches from a dataset using a sampler and collator.

    Args:
        data: The dataset to iterate over.
        sampler: Batch sampler that generates batch indices.
        collator: Function that collates a sequence of items into a batch.

    Examples:
        >>> def collator(batch):
        ...     return [x * 2 for x in batch]
        >>> sampler = BasicBatchSampler(batch_size=2)
        >>> iterator = BatchIterator([1, 2, 3, 4], sampler, collator)
        >>> list(iterator)
        [[2, 4], [6, 8]]
    """

    def __init__(
        self,
        data: Sequence[_InputT],
        sampler: BaseBatchSampler,
        collator: Callable[[Sequence[_InputT]], _BatchT],
    ) -> None:
        self._data = data
        self._collator = collator
        self._sampler = sampler
        self._iterator: Iterator[Sequence[int]] | None = None

    def __iter__(self) -> Iterator[_BatchT]:
        return self

    def __next__(self) -> _BatchT:
        if self._iterator is None:
            self._iterator = iter(self._sampler(self._data))
        batch_indices = next(self._iterator)
        return self._collator([self._data[i] for i in batch_indices])


class DataLoader(Generic[_InputT, _BatchT]):
    """Data loader that creates batched iterators from datasets.

    The DataLoader combines a batch sampler and a collator function to create
    batched data iterators. It optionally supports buffering using a separate
    process to prefetch batches in the background, which can improve throughput
    when batch collation is expensive.

    Args:
        sampler: Batch sampler that generates batch indices.
        collator: Function that collates a sequence of items into a batch.
        buffer_size: Size of the prefetch buffer. If 0, buffering is disabled.
            If > 0, batches are prepared in a background process. Defaults to 0.

    Examples:
        Basic usage without buffering:
        >>> def collator(batch):
        ...     return [x * 2 for x in batch]
        >>> sampler = BasicBatchSampler(batch_size=4, shuffle=True)
        >>> loader = DataLoader(sampler=sampler, collator=collator)
        >>> dataset = list(range(10))
        >>> batches = list(loader(dataset))

        With buffering for better performance:
        >>> loader = DataLoader(
        ...     sampler=sampler,
        ...     collator=collator,
        ...     buffer_size=10,  # Prefetch up to 10 batches
        ... )
        >>> from formed.common.ctxutils import closing
        >>> with closing(loader(dataset)) as batches:
        ...     for batch in batches:
        ...         # Process batch
        ...         pass

    Note:
        When using buffering (buffer_size > 0), the collator function and
        any objects it references must be picklable, as they will be passed
        to a background process. Also, it's recommended to use the loader
        with a context manager (closing) to ensure proper cleanup of the
        background process.
    """

    def __init__(
        self,
        sampler: BaseBatchSampler,
        collator: Callable[[Sequence[_InputT]], _BatchT],
        buffer_size: int = 0,
    ) -> None:
        self._collator = collator
        self._sampler = sampler
        self._buffer_size = buffer_size

    def __call__(self, data: Sequence[_InputT]) -> SizedIterator[_BatchT]:
        """Create a batched iterator for the given dataset.

        Args:
            data: The dataset to create batches from.

        Returns:
            A sized iterator that yields batches. If buffer_size > 0, the
            iterator will prefetch batches in a background process.
        """
        iterator = BatchIterator(data, self._sampler, self._collator)
        if self._buffer_size > 0:
            iterator = BufferedIterator[_BatchT](iterator, buffer_size=self._buffer_size)
        return SizedIterator(iterator, len(self._sampler(data)))
