import dataclasses
from collections.abc import Sequence
from typing import Any

from typing_extensions import TypeVar

import formed.integrations.ml as ml
import formed.integrations.ml.types as mlt


InputT = TypeVar(
    "InputT",
    default=Any,
)
TextTransformT_co = TypeVar(
    "TextTransformT_co",
    bound=ml.BaseTransform[str | Sequence[str] | mlt.AnalyzedText],
    default=Any,
    covariant=True,
)


@dataclasses.dataclass
class ClassificationExample:
    id: str
    text: str | Sequence[str]
    label: int | str | None = None
