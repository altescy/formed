import math
import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Optional

from collatable import Collator
from colt import Registrable

from formed.common.iterutils import SizedIterator

from .fields import Field
from .types import DataArray


class BatchSampler(Registrable):
    def __call__(self, data: Sequence) -> SizedIterator[Sequence[int]]:
        raise NotImplementedError


@BatchSampler.register("basic")
class BasicBatchSampler(BatchSampler):
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


class DataLoader:
    def __init__(
        self,
        batch_sampler: BatchSampler,
        collator: Optional[Collator] = None,
    ) -> None:
        self._batch_sampler = batch_sampler
        self._collator = collator or Collator()

    def __call__(self, data: Sequence[Mapping[str, Field]]) -> SizedIterator[Mapping[str, DataArray]]:
        indices = self._batch_sampler(data)
        return SizedIterator(
            (self._collator([data[i] for i in batch]) for batch in indices),
            len(indices),
        )
