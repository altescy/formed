import dataclasses
from typing import Any, ClassVar, Protocol, TypeVar, Union, runtime_checkable

from pydantic import BaseModel, JsonValue

__all__ = [
    "DataContainer",
    "IDataclass",
    "INamedTuple",
    "JsonValue",
    "S_DataContainer",
    "T_DataContainer",
    "T_NamedTuple",
]


@runtime_checkable
class IDataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field]]


@runtime_checkable
class INamedTuple(Protocol):
    _fields: ClassVar[tuple[str, ...]]

    def _asdict(self) -> dict[str, Any]: ...

    def _replace(self: "T_NamedTuple", **kwargs: Any) -> "T_NamedTuple": ...


@runtime_checkable
class IJsonSerializable(Protocol):
    def json(self) -> JsonValue: ...


DataContainer = Union[IDataclass, INamedTuple, BaseModel, dict[str, Any]]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

T_NamedTuple = TypeVar("T_NamedTuple", bound=INamedTuple)

S_DataContainer = TypeVar("S_DataContainer", bound=DataContainer)
T_DataContainer = TypeVar("T_DataContainer", bound=DataContainer)
