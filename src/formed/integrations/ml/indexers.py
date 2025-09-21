import json
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from os import PathLike
from typing import Generic, Optional, TypeVar, Union

import numpy

from .types import Tensor

HashableT = TypeVar("HashableT", bound=Hashable)
IndexerT = TypeVar("IndexerT", bound="Indexer")


class Indexer(Generic[HashableT], Mapping[HashableT, int]):
    def __init__(
        self,
        *,
        ignores: Iterable[HashableT] = (),
        reserved: Sequence[HashableT] = (),
        default: Optional[HashableT] = None,
        min_count: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> None:
        if default is not None and default in ignores:
            raise ValueError(f"default value {default} is in ignores")
        if default is not None and default not in reserved:
            raise ValueError(f"default value {default} is not in reserved")

        self._ignores = set(ignores)
        self._reserved = reserved
        self._default = default
        self._min_count = min_count
        self._max_size = max_size

        self._frozen = False

        self._value_to_index: dict[HashableT, int] = {}
        self._index_to_value: dict[int, HashableT] = {}
        self._count = Counter[HashableT]()

        for value in reserved:
            if value in self._value_to_index:
                raise ValueError(f"reserved value {value} is duplicated")
            index = len(self._value_to_index)
            self._value_to_index[value] = index
            self._index_to_value[index] = value

    @property
    def default(self) -> Optional[HashableT]:
        return self._default

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        if self._frozen:
            return
        self._frozen = True

        values_to_remove = set[HashableT]()

        if self._min_count is not None:
            for value, count in self._count.items():
                if count < self._min_count:
                    values_to_remove.add(value)
        if self._max_size is not None:
            values = sorted(self._count.keys(), key=lambda value: self._count[value], reverse=True)
            for value in values[self._max_size :]:
                values_to_remove.add(value)

        if values_to_remove:
            value_to_index: dict[HashableT, int] = {}
            index_to_value: dict[int, HashableT] = {}
            for value in values_to_remove:
                index = self._value_to_index.pop(value)
                del self._index_to_value[index]
                del self._count[value]
            for value, index in self._value_to_index.items():
                index = len(value_to_index)
                value_to_index[value] = index
                index_to_value[index] = value
            self._value_to_index = value_to_index
            self._index_to_value = index_to_value

    def add(self, value: HashableT, /) -> None:
        if self._frozen:
            raise RuntimeError("indexer is frozen")
        if value in self._ignores:
            return
        if value not in self._value_to_index:
            index = len(self._value_to_index)
            self._value_to_index[value] = index
            self._index_to_value[index] = value
        self._count[value] += 1

    def get_index_by_value(self, value: HashableT, /) -> int:
        if not self._frozen:
            raise RuntimeError("indexer is not frozen")
        if self._default is not None and value in self._ignores:
            return self._value_to_index[self._default]
        if value not in self._value_to_index:
            if self._default is None:
                raise KeyError(value)
            return self._value_to_index[self._default]
        return self._value_to_index[value]

    def get_value_by_index(self, index: int, /) -> HashableT:
        if not self._frozen:
            raise RuntimeError("indexer is not frozen")
        return self._index_to_value[index]

    def __len__(self) -> int:
        return len(self._value_to_index)

    def __getitem__(self, value: HashableT, /) -> int:
        return self.get_index_by_value(value)

    def __iter__(self) -> Iterator[HashableT]:
        return iter(self._value_to_index)

    def __contains__(self, value: object, /) -> bool:
        return value in self._value_to_index

    def save(self, path: Union[str, PathLike], /) -> None:
        if not self._frozen:
            raise RuntimeError("indexer is not frozen")
        with open(path, "w") as jsonfile:
            json.dump(
                {
                    "default": self._default,
                    "reserved": list(self._reserved),
                    "value_to_index": [[value, index] for value, index in self._value_to_index.items()],
                },
                jsonfile,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls: type[IndexerT], path: Union[str, PathLike], /) -> IndexerT:
        with open(path, "r") as jsonfile:
            data = json.load(jsonfile)
        indexer = cls(default=data["default"], reserved=data["reserved"])
        for value, index in data["value_to_index"]:
            indexer._value_to_index[value] = index
            indexer._index_to_value[index] = value
        indexer._frozen = True
        return indexer


class LabelIndexer(Generic[HashableT], Indexer[HashableT]):
    def encode(self, value: HashableT, /) -> int:
        return self.get_index_by_value(value)

    def decode(self, index: int, /) -> HashableT:
        return self.get_value_by_index(index)

    def __call__(self, value: HashableT, /) -> int:
        return self.encode(value)


class TokenIndexer(Generic[HashableT], Indexer[HashableT]):
    def encode(self, values: Sequence[HashableT], /) -> dict[str, Tensor]:
        token_ids = numpy.array([self.get_index_by_value(value) for value in values], dtype=numpy.int32)
        mask = numpy.ones_like(token_ids, dtype=numpy.bool_)
        return {"token_ids": token_ids, "mask": mask}

    def decode(self, index: Mapping[str, Tensor], /) -> Sequence[HashableT]:
        token_ids = index["token_ids"]
        mask = index["mask"]
        return [self.get_value_by_index(token_id) for token_id, m in zip(token_ids, mask) if m]

    def __call__(self, values: Sequence[HashableT], /) -> dict[str, Tensor]:
        return self.encode(values)
