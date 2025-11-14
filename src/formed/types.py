import dataclasses
from typing import Any, ClassVar, Literal, Protocol, TypeVar, Union, runtime_checkable

from typing_extensions import Self, TypeAlias

__all__ = [
    "DataContainer",
    "IDataclass",
    "INamedTuple",
    "JsonValue",
    "S_DataContainer",
    "T_DataContainer",
    "T_NamedTuple",
]

JsonValue: TypeAlias = Union[
    list["JsonValue"],
    dict[str, "JsonValue"],
    str,
    bool,
    int,
    float,
    None,
]


@runtime_checkable
class IDataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field]]


@runtime_checkable
class INamedTuple(Protocol):
    _fields: ClassVar[tuple[str, ...]]
    _field_defaults: ClassVar[dict[str, Any]]

    def _asdict(self) -> dict[str, Any]: ...

    def _replace(self: "T_NamedTuple", **kwargs: Any) -> "T_NamedTuple": ...


@runtime_checkable
class IJsonSerializable(Protocol):
    def json(self) -> JsonValue: ...


@runtime_checkable
class IPydanticModel(Protocol):
    def model_dump(self, *, mode: Literal["json", "python"] = "python") -> dict[str, Any]: ...

    @classmethod
    def model_validate(cls, obj: Any) -> Self: ...


DataContainer = Union[IDataclass, INamedTuple, IPydanticModel, dict[str, Any]]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

T_NamedTuple = TypeVar("T_NamedTuple", bound=INamedTuple)

S_DataContainer = TypeVar("S_DataContainer", bound=DataContainer)
T_DataContainer = TypeVar("T_DataContainer", bound=DataContainer)
