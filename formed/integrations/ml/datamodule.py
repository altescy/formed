import copy
import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from types import TracebackType
from typing import Any, Generic, Optional, TypeVar, Union, cast

from formed.common.xutils import xgetattr

from .fields import Field
from .transforms import FieldTransform
from .types import DataArray

S = TypeVar("S")
T = TypeVar("T")
FieldT = TypeVar("FieldT", bound=Field)

_FORMEDML_DATAMODULE = ContextVar[Optional["DataModule"]]("formedml::datamodule", default=None)


class FieldAccessor:
    def __init__(self, field: str) -> None:
        self._field = field

    def __call__(self, obj: Any) -> Any:
        return xgetattr(obj, self._field)


@dataclasses.dataclass(frozen=True)
class FieldConfig(Generic[S, T, FieldT]):
    accessor: Union[str, Callable[[S], T]]
    transformer: Union[Callable[[], FieldTransform[T, FieldT]], FieldTransform[T, FieldT]]
    is_optional: bool = False

    @property
    def access(self) -> Callable[[S], T]:
        return cast(Callable[[S], T], FieldAccessor(self.accessor)) if isinstance(self.accessor, str) else self.accessor

    @property
    def transform(self) -> FieldTransform[T, FieldT]:
        return self.transformer if isinstance(self.transformer, FieldTransform) else self.transformer()

    @property
    def reconstruct(self) -> Callable[[DataArray], T]:
        return self.transform.reconstruct

    @property
    def stats(self) -> Mapping[str, Any]:
        return self.transform.stats()


class DataModuleNotActivatedError(RuntimeError):
    pass


class DataModule(Generic[T]):
    def __init__(
        self,
        fields: Optional[Mapping[str, Union[FieldTransform, FieldConfig]]] = None,
    ) -> None:
        self._fields: Optional[Mapping[str, FieldConfig]] = (
            {
                key: (
                    FieldConfig(
                        accessor=FieldAccessor(key),
                        transformer=value,
                    )
                    if isinstance(value, FieldTransform)
                    else value
                )
                for key, value in fields.items()
            }
            if fields is not None
            else None
        )
        self._is_built = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fields={self._fields})"

    def build(self, dataset: Sequence[T]) -> None:
        if self._is_built:
            return
        if self._fields is None:
            self._fields = _infer_fields(type(dataset[0]))
        for field in self._fields.values():
            field.transform.build(
                (target for item in dataset if (target := field.access(item)) is not None or not field.is_optional)
            )
        for field in self._fields.values():
            field.transform.freeze()
        self._is_built = True

    def activate(self) -> None:
        set_datamodule(self)

    def deactivate(self) -> None:
        if not self.is_active():
            raise DataModuleNotActivatedError("This DataModule is not currently active")
        unset_datamodule()

    def is_active(self) -> bool:
        try:
            return use_datamodule() is self
        except DataModuleNotActivatedError:
            return False

    def field(self, key: str) -> FieldConfig:
        if self._fields is None:
            raise RuntimeError("Fields are not defined")
        return self._fields[key]

    def __enter__(self) -> "DataModule":
        if not self.is_active():
            self.activate()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.deactivate()

    def __call__(self, dataset: Iterable[T]) -> Iterator[dict[str, Field]]:
        if self._fields is None:
            raise RuntimeError("Fields are not defined")
        for item in dataset:
            yield {
                key: field.transform(target)
                for key, field in self._fields.items()
                if (target := field.access(item)) is not None or not field.is_optional
            }


def use_datamodule() -> "DataModule":
    datamodule = _FORMEDML_DATAMODULE.get()
    if datamodule is None:
        raise DataModuleNotActivatedError("No DataModule is currently active")
    return datamodule


def set_datamodule(datamodule: "DataModule") -> None:
    _FORMEDML_DATAMODULE.set(datamodule)


def unset_datamodule() -> None:
    _FORMEDML_DATAMODULE.set(None)


def extract_fields(cls: type) -> Optional[Mapping[str, FieldConfig]]:
    import typing

    from colt.utils import reveal_origin

    fields: dict[str, FieldConfig] = {}
    for key, value in typing.get_type_hints(cls, include_extras=True).items():
        if reveal_origin(value) == typing.Annotated:
            field_config = next(
                filter(lambda x: isinstance(x, FieldConfig), typing.get_args(value)),
                None,
            )
            if field_config is not None:
                field_config = copy.deepcopy(field_config)
                if not isinstance(field_config.transformer, FieldTransform):
                    field_config = dataclasses.replace(field_config, transformer=field_config.transformer())
                fields[key] = field_config

    return fields or None


def _infer_fields(cls: type) -> Mapping[str, FieldConfig]:
    import typing
    from collections.abc import Hashable

    import numpy
    from colt.utils import issubtype, remove_optional

    from .transforms import (
        LabelFieldTransform,
        ListFieldTransform,
        ScalarFieldTransform,
        TensorFieldTransform,
        TextFieldTransform,
    )
    from .types import Scalar

    def check_optional(annotation: typing.Any) -> bool:
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        return origin == typing.Union and type(None) in args

    fields: dict[str, FieldConfig] = {}
    for key, value in typing.get_type_hints(cls).items():
        is_optional = check_optional(value)
        value = remove_optional(value)
        if issubtype(value, Union[str, Sequence[str]]):
            fields[key] = FieldConfig(
                accessor=FieldAccessor(value),
                transformer=TextFieldTransform(),
                is_optional=is_optional,
            )
        elif issubtype(value, Scalar):
            fields[key] = FieldConfig(
                accessor=FieldAccessor(key),
                transformer=ScalarFieldTransform(),
                is_optional=is_optional,
            )
        elif isinstance(value, type) and issubclass(value, numpy.ndarray):
            fields[key] = FieldConfig(
                accessor=FieldAccessor(key),
                transformer=TensorFieldTransform(),
                is_optional=is_optional,
            )
        elif issubtype(value, Sequence[Hashable]):
            fields[key] = FieldConfig(
                accessor=FieldAccessor(key),
                transformer=ListFieldTransform(
                    transform=LabelFieldTransform(),
                ),
                is_optional=is_optional,
            )
        elif typing.get_origin(value) == typing.Literal:
            fields[key] = FieldConfig(
                accessor=FieldAccessor(key),
                transformer=LabelFieldTransform(),
                is_optional=is_optional,
            )
        else:
            raise ValueError(f"Cannnot infer field type for {key}: {value}")

    return fields
