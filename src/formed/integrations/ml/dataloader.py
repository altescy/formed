import abc
import math
import random
from collections.abc import Callable, Iterator, Sequence
from typing import Generic, TypeVar

from colt import Registrable

from formed.common.iterutils import SizedIterator

_InputT = TypeVar("_InputT")
_BatchT = TypeVar("_BatchT")


class BaseBatchSampler(Registrable, abc.ABC):
    @abc.abstractmethod
    def __call__(self, data: Sequence) -> SizedIterator[Sequence[int]]:
        raise NotImplementedError


@BaseBatchSampler.register("basic")
class BasicBatchSampler(BaseBatchSampler):
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


class DataLoader(Generic[_InputT, _BatchT]):
    def __init__(
        self,
        sampler: Callable[[Sequence[_InputT]], SizedIterator[Sequence[int]]],
        collator: Callable[[Sequence[_InputT]], _BatchT],
    ) -> None:
        self._collator = collator
        self._sampler = sampler

    def __call__(self, data: Sequence[_InputT]) -> SizedIterator[_BatchT]:
        indices = self._sampler(data)
        return SizedIterator(
            (self._collator([data[i] for i in batch]) for batch in indices),
            len(indices),
        )
