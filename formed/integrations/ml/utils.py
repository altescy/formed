import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import ClassVar, Optional, Pattern

from collatable.utils import debatched  # noqa: F401


class RegexTokenizer:
    _DEFAULT_TOKENIZER_PATTERN: ClassVar[Pattern] = re.compile(r"[^\s.,!?:;/]+(?:[-']\[^\s.,!?:;/]+)*|[.,!?:;/]")

    def __init__(self, pattern: Optional[Pattern] = None) -> None:
        self._token_pattern = pattern or self._DEFAULT_TOKENIZER_PATTERN

    def __call__(self, text: str) -> list[str]:
        return self._token_pattern.findall(text)


class MetricAverage(Mapping[str, float]):
    def __init__(self) -> None:
        self._total: dict[str, float] = defaultdict(float)
        self._count = 0

    def add(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self._total[key] += value
        self._count += 1

    def reset(self) -> None:
        self._total.clear()
        self._count = 0

    def __getitem__(self, key: str) -> float:
        return self._total[key] / self._count

    def __iter__(self) -> Iterator[str]:
        return iter(self._total)

    def __len__(self) -> int:
        return len(self._total)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self)})"
