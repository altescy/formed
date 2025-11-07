import dataclasses
import sys
import typing
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any, ClassVar, Generic, Optional, Union, cast

from typing_extensions import TypeVar

from ..types import AsBatch, AsConverter, AsInstance, DataModuleMode
from .base import BaseTransform, Extra, Param

if sys.version_info >= (3, 10):
    from types import UnionType
else:

    class UnionType: ...


_DataModuleModeT_co = TypeVar("_DataModuleModeT_co", bound=DataModuleMode, covariant=True)


_T = TypeVar("_T", default=Any)
_InstanceT = TypeVar("_InstanceT", bound="DataModule[AsInstance]", default=Any)
_BatchT = TypeVar("_BatchT", bound="DataModule[AsBatch]", default=Any)


class _Unavailable: ...


_UNAVAILABLE = _Unavailable()


def _is_param_field(annotation: Any) -> bool:
    if annotation is Param:
        return True
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (Union, UnionType) and args:
        return any(_is_param_field(arg) for arg in args)
    return False


def _is_extra_field(annotation: Any) -> bool:
    if annotation is Extra:
        return True
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (Union, UnionType) and args:
        return any(_is_extra_field(arg) for arg in args)
    return False


class DataModule(
    BaseTransform[_T, _T, _InstanceT, _BatchT],
    Generic[_DataModuleModeT_co, _T, _InstanceT, _BatchT],
):
    __mode__: Optional[_DataModuleModeT_co] = dataclasses.field(default=None, init=False, repr=False, compare=False)
    __param_fields__: ClassVar[Optional[Mapping[str, dataclasses.Field]]] = None
    __extra_fields__: ClassVar[Optional[Mapping[str, dataclasses.Field]]] = None

    _batch_size: Optional[int] = dataclasses.field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def __get_param_fields__(cls) -> Mapping[str, dataclasses.Field]:
        if cls.__param_fields__ is None:
            cls.__param_fields__ = {
                field.name: field for field in dataclasses.fields(cls) if _is_param_field(field.type)
            }
        return cls.__param_fields__

    @classmethod
    def __get_extra_fields__(cls) -> Mapping[str, dataclasses.Field]:
        if cls.__extra_fields__ is None:
            cls.__extra_fields__ = {
                field.name: field for field in dataclasses.fields(cls) if _is_extra_field(field.type)
            }
        return cls.__extra_fields__

    def __post_init__(self) -> None:
        if self.__mode__ not in (None, DataModuleMode.AS_CONVERTER):
            return
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, BaseTransform):
                value.__set_name__(self.__class__, field.name)

    @property
    def __field_transforms__(self) -> Mapping[str, BaseTransform]:
        assert self.__mode__ in (None, DataModuleMode.AS_CONVERTER), (
            "Field transforms are only available in converter mode"
        )
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if isinstance(getattr(self, field.name), BaseTransform)
        }

    @contextmanager
    def train(self) -> Iterator[None]:
        assert self.__mode__ in (None, DataModuleMode.AS_CONVERTER), (
            "DataModule must be in converter mode to enter training mode"
        )
        with ExitStack() as stack:
            for transform in self.__field_transforms__.values():
                stack.enter_context(transform.train())
            yield

    def instance(self: "DataModule[AsConverter]", obj: _T, /) -> _InstanceT:
        assert self.__mode__ in (None, DataModuleMode.AS_CONVERTER), (
            "DataModule must be in converter mode to create an instance"
        )

        fields = {}
        for name, transform in self.__field_transforms__.items():
            fields[name] = transform(obj)
        for name, field in self.__class__.__get_param_fields__().items():
            if (
                name not in fields
                and field.default is not dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                fields[name] = _UNAVAILABLE

        instance = cast(_InstanceT, dataclasses.replace(self, **fields))
        setattr(instance, "__mode__", DataModuleMode.AS_INSTANCE)

        return instance

    def batch(self: "DataModule[AsConverter]", instances: Sequence[Union[_T, _InstanceT]]) -> _BatchT:
        assert self.__mode__ in (None, DataModuleMode.AS_CONVERTER), (
            "DataModule must be in converter mode to create a batch"
        )

        instances = [item if isinstance(item, DataModule) else self.instance(item) for item in instances]
        fields = {}
        for name, transform in self.__field_transforms__.items():
            can_be_optional = name in self.__class__.__get_extra_fields__()
            values = [getattr(instance, name) for instance in instances]
            if can_be_optional and all(value is None for value in values):
                fields[name] = None
            else:
                fields[name] = transform.batch(values)
        for name in self.__class__.__get_param_fields__().keys():
            if name not in fields:
                fields[name] = _UNAVAILABLE

        batch = cast(_BatchT, dataclasses.replace(self, **fields))
        setattr(batch, "__mode__", DataModuleMode.AS_BATCH)

        batch._batch_size = len(instances)
        return batch

    def __call__(self, data: Union[_T, _InstanceT], /) -> Optional[_InstanceT]:
        if isinstance(data, self.__class__) and data.__mode__ == DataModuleMode.AS_INSTANCE:
            return cast(Optional[_InstanceT], data)
        return super().__call__(cast(_T, data))

    def _get_input_value(self, data: _T) -> Optional[_T]:
        if self._parent is None:
            return data
        return super()._get_input_value(data)

    def __len__(self: "DataModule[AsBatch]") -> int:
        assert self.__mode__ == DataModuleMode.AS_BATCH, "Length is only available in batch mode"
        assert self._batch_size is not None, "Batch size is not set"
        return self._batch_size

    def __getstate__(self) -> Mapping[str, Any]:
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        setattr(self, "__mode__", state.get("__mode__", None))
        for field in dataclasses.fields(self):
            if field.name in state:
                setattr(self, field.name, state[field.name])
