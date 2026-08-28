import dataclasses
from collections.abc import Callable, Mapping
from typing import Generic, TypedDict, TypeVar

from formed.types import JsonValue

ResultT = TypeVar("ResultT")


class ToolSpec(TypedDict):
    name: str
    description: str
    parameters: Mapping[str, JsonValue]


@dataclasses.dataclass(frozen=True)
class ToolDefinition(Generic[ResultT]):
    spec: ToolSpec
    func: Callable[..., ResultT]

    @property
    def name(self) -> str:
        return self.spec["name"]
