import dataclasses
import operator
from collections.abc import Sequence
from contextlib import suppress
from logging import getLogger
from typing import Any, Generic

import numpy
from typing_extensions import TypeVar

from ..types import LabelT
from .base import BaseTransform

logger = getLogger(__name__)


_S = TypeVar("_S", default=Any)
_T = TypeVar("_T", default=Any)


@BaseTransform.register("metadata")
class MetadataTransform(
    Generic[_S, _T],
    BaseTransform[_S, _T, _T, Sequence[_T]],
):
    def instance(self, value: _T, /) -> _T:
        return value

    def batch(self, batch: Sequence[_T], /) -> Sequence[_T]:
        return list(batch)


@BaseTransform.register("label")
class LabelIndexer(BaseTransform[_S, LabelT, int, numpy.ndarray], Generic[_S, LabelT]):
    label2id: Sequence[tuple[LabelT, int]] = dataclasses.field(default_factory=list)
    freeze: bool = dataclasses.field(default=False)

    _label_counts: list[tuple[LabelT, int]] = dataclasses.field(
        default_factory=list, init=False, repr=False, compare=False
    )

    @property
    def num_labels(self) -> int:
        return len(self.label2id)

    @property
    def labels(self) -> list[LabelT]:
        return [label for label, _ in sorted(self.label2id, key=operator.itemgetter(1))]

    @property
    def occurrences(self) -> dict[LabelT, int]:
        return dict(self._label_counts)

    @property
    def distribution(self) -> numpy.ndarray:
        total = sum(count for _, count in self._label_counts) + self.num_labels
        counts = [count for _, count in sorted(self._label_counts, key=operator.itemgetter(1))]
        return numpy.array([(count + 1) / total for count in counts], dtype=numpy.float32)

    def _on_start_training(self) -> None:
        self._label_counts.clear()

    def _on_end_training(self) -> None:
        pass

    def get_index(self, value: LabelT, /) -> int:
        with suppress(StopIteration):
            return next(label_id for label, label_id in self.label2id if label == value)
        raise KeyError(value)

    def get_value(self, index: int, /) -> LabelT:
        for label, label_id in self.label2id:
            if label_id == index:
                return label
        raise KeyError(index)

    def ingest(self, value: LabelT, /) -> None:
        if self.freeze:
            return
        if self._training:
            try:
                self.get_index(value)
            except KeyError:
                self.label2id = list(self.label2id) + [(value, len(self.label2id))]
                self._label_counts.append((value, 0))
            for index, (label, count) in enumerate(self._label_counts):
                if label == value:
                    self._label_counts[index] = (label, count + 1)
                    break
        else:
            logger.warning("Ignoring ingest call when not in training mode")

    def instance(self, label: LabelT, /) -> int:
        if self._training:
            self.ingest(label)
        return self.get_index(label)

    def batch(self, batch: Sequence[int], /) -> numpy.ndarray:
        return numpy.array(batch, dtype=numpy.int64)

    def reconstruct(self, batch: numpy.ndarray, /) -> list[LabelT]:
        return [self.get_value(index) for index in batch.tolist()]


@BaseTransform.register("scalar")
class ScalarTransform(
    Generic[_S],
    BaseTransform[_S, float, float, numpy.ndarray],
):
    def instance(self, value: float, /) -> float:
        return value

    def batch(self, batch: Sequence[float], /) -> numpy.ndarray:
        return numpy.array(batch, dtype=numpy.float32)


@BaseTransform.register("tensor")
class TensorTransform(
    Generic[_S],
    BaseTransform[_S, numpy.ndarray, numpy.ndarray, numpy.ndarray],
):
    def instance(self, value: numpy.ndarray, /) -> numpy.ndarray:
        return value

    def batch(self, batch: Sequence[numpy.ndarray], /) -> numpy.ndarray:
        return numpy.stack(batch, axis=0)
