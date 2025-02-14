import math
import random
from collections.abc import Callable, Iterator, Sequence
from typing import Generic, TypeVar

from formed.common.iterutils import SizedIterator

from .types import IBatchIterator, ModelInputT

DataT = TypeVar("DataT")


class DataLoader(Generic[DataT, ModelInputT]):
    def __init__(
        self,
        collator: Callable[[Sequence[DataT]], ModelInputT],
        batch_size: int = 32,
        drop_last: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self._collator = collator
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._shuffle = shuffle
        self._rng = random.Random(seed)

    def __call__(self, dataset: Sequence[DataT]) -> IBatchIterator[ModelInputT]:
        def iterator() -> Iterator[ModelInputT]:
            indices = list(range(len(dataset)))
            if self._shuffle:
                self._rng.shuffle(indices)
            batch: list[DataT] = []
            for i in indices:
                batch.append(dataset[i])
                if len(batch) == self._batch_size:
                    yield self._collator(batch)
                    batch = []
            if batch and not self._drop_last:
                yield self._collator(batch)

        return SizedIterator(
            iterator(),
            size=len(dataset) // self._batch_size if self._drop_last else math.ceil(len(dataset) / self._batch_size),
        )
