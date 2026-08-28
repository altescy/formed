import dataclasses
from typing import Generic, TypeVar, Union

from typing_extensions import TypeAlias

ResultT = TypeVar("ResultT")


@dataclasses.dataclass(frozen=True)
class Continue: ...


@dataclasses.dataclass(frozen=True)
class Stop(Generic[ResultT]):
    result: ResultT


Control: TypeAlias = Union[Continue, Stop[ResultT]]
