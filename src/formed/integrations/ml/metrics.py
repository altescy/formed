from collections import defaultdict
from collections.abc import Mapping
from typing import Generic, TypeVar

from colt import Registrable

_T = TypeVar("_T")


class BaseMetric(Registrable, Generic[_T]):
    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        raise NotImplementedError()

    def update(self, inputs: _T) -> None:
        raise NotImplementedError()

    def compute(self) -> dict[str, float]:
        raise NotImplementedError()


@BaseMetric.register("empty")
class EmptyMetric(BaseMetric[_T]):
    def reset(self) -> None:
        pass

    def update(self, inputs: _T) -> None:
        pass

    def compute(self) -> dict[str, float]:
        return {}


@BaseMetric.register("average")
class AverageMetric(BaseMetric[Mapping[str, float]]):
    def __init__(self) -> None:
        self._total: dict[str, float] = defaultdict(float)
        self._count = 0

    def reset(self) -> None:
        self._total = defaultdict(float)
        self._count = 0

    def update(self, inputs: Mapping[str, float]) -> None:
        for key, value in inputs.items():
            self._total[key] += value
        self._count += 1

    def compute(self) -> dict[str, float]:
        return {key: value / self._count for key, value in self._total.items()}
