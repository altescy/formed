import dataclasses
from typing import Any, ClassVar, Literal, Protocol, TypeVar, Union, runtime_checkable

from typing_extensions import Self, TypeAlias

from formed.common.singleton import BaseSingleton

__all__ = [
    "DataContainer",
    "IDataclass",
    "INamedTuple",
    "JsonValue",
    "DataContainerS",
    "DataContainerT",
    "NamedTupleT",
    "SupportsClosing",
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


class NotSpecified(BaseSingleton): ...


@runtime_checkable
class SupportsClosing(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class IDataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field]]


@runtime_checkable
class INamedTuple(Protocol):
    _fields: ClassVar[tuple[str, ...]]
    _field_defaults: ClassVar[dict[str, Any]]

    def _asdict(self) -> dict[str, Any]: ...

    def _replace(self: "NamedTupleT", **kwargs: Any) -> "NamedTupleT": ...


@runtime_checkable
class IJsonSerializable(Protocol):
    def json(self) -> JsonValue: ...


@runtime_checkable
class IJsonDeserializable(Protocol):
    @classmethod
    def from_json(cls, o: JsonValue, /) -> Self: ...


@runtime_checkable
class IJsonCompatible(IJsonSerializable, IJsonDeserializable, Protocol): ...


@runtime_checkable
class IPydanticModel(Protocol):
    def model_dump(self, *, mode: Literal["json", "python"] = "python") -> dict[str, Any]: ...

    @classmethod
    def model_validate(cls, obj: Any) -> Self: ...


DataContainer = Union[IDataclass, INamedTuple, IPydanticModel, dict[str, Any]]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

NamedTupleT = TypeVar("NamedTupleT", bound=INamedTuple)

DataContainerS = TypeVar("DataContainerS", bound=DataContainer)
DataContainerT = TypeVar("DataContainerT", bound=DataContainer)
