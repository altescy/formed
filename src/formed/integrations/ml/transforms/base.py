import abc
import dataclasses
import sys
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, suppress
from functools import partial
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Final, Generic, Literal, Optional, Union, cast, overload

import cloudpickle
from colt import Registrable
from typing_extensions import Self, TypeVar, dataclass_transform

from formed.common.attributeutils import xgetattr
from formed.common.jax import JAX_STATIC_FIELD

from ..types import (
    AsBatch,
    AsConverter,
    AsInstance,
    BatchT,
    BatchT_co,
    DataModuleMode,
    InstanceT,
    InstanceT_co,
)

if sys.version_info >= (3, 10):
    from types import UnionType
else:

    class UnionType: ...


logger = getLogger(__name__)


_S = TypeVar("_S", default=Any)
_T = TypeVar("_T", default=Any)
_T_co = TypeVar("_T_co", covariant=True)
_TypeT = TypeVar("_TypeT", bound=type)
_BaseTransformT = TypeVar("_BaseTransformT", bound="BaseTransform")
_BaseTransformT_co = TypeVar("_BaseTransformT_co", bound="BaseTransform", covariant=True)


_DATACLASS_REGISTRY: Final = set[type]()


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


def _find_dataclass_field(annotation: Any) -> Optional[type]:
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (Union, UnionType) and args:
        for arg in args:
            result = _find_dataclass_field(arg)
            if result is not None:
                return result
    return None


def register_dataclass(cls: _TypeT) -> _TypeT:
    if cls in _DATACLASS_REGISTRY:
        return cls

    _DATACLASS_REGISTRY.add(cls)

    with suppress(ImportError):
        import jax

        if getattr(cls, "__is_datamodule__", False):
            for field in dataclasses.fields(cls):
                field_class = _find_dataclass_field(field.type)
                if field_class is not None:
                    register_dataclass(field_class)

        drop_fields = [f.name for f in dataclasses.fields(cls) if not f.init and not _is_param_field(f.type)]
        data_fields = [
            f.name
            for f in dataclasses.fields(cls)
            if not f.metadata.get(JAX_STATIC_FIELD, False) and f.name not in drop_fields
        ]
        meta_fields = [
            f.name
            for f in dataclasses.fields(cls)
            if f.metadata.get(JAX_STATIC_FIELD, False) and f.name not in drop_fields
        ]

        try:
            jax.tree_util.register_dataclass(
                cls,
                data_fields=data_fields,
                meta_fields=meta_fields,
                drop_fields=drop_fields,
            )
        except ValueError as error:
            if str(error.args[0]).startswith("Duplicate custom dataclass"):
                pass
            else:
                raise

    return cls


class Extra(Generic[_BaseTransformT_co]):
    @classmethod
    def __class_getitem__(cls, item: type["BaseTransform"]) -> Any:
        return Union[Optional[item], cls]

    @classmethod
    def default(
        cls: type["Extra[_BaseTransformT]"],
        default: Optional[_BaseTransformT] = None,
    ) -> "Extra[_BaseTransformT]":
        return cast(Extra[_BaseTransformT], default)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("Extra is a marker class and cannot be instantiated directly")

    def __set__(
        self: "Extra[_BaseTransformT]",
        instance: "DataModule",
        value: Optional[_BaseTransformT],
    ) -> None: ...

    @overload
    def __get__(
        self: "Extra[BaseTransform[Any, Any, Any, BatchT]]",
        instance: "DataModule[AsBatch]",
        owner: type["DataModule[AsBatch]"],
    ) -> Optional[BatchT]: ...

    @overload
    def __get__(
        self: "Extra[BaseTransform[Any, Any, InstanceT, Any]]",
        instance: "DataModule[AsInstance]",
        owner: type["DataModule[AsInstance]"],
    ) -> Optional[InstanceT]: ...

    @overload
    def __get__(
        self,
        instance: "DataModule[AsConverter]",
        owner: type["DataModule[AsConverter]"],
    ) -> _BaseTransformT_co: ...

    def __get__(
        self,
        instance: "DataModule",
        owner: type["DataModule"],
    ) -> Union[_BaseTransformT_co, Any]: ...


class Param(Generic[_T_co]):
    @classmethod
    def __class_getitem__(cls: type["Param[_T]"], item: type[_T]) -> Any:
        return Union[item, cls]

    @classmethod
    def default(cls: type["Param[_T]"], default: _T) -> "Param[_T]":
        return cast(Param[_T], default)

    @classmethod
    def default_factory(
        cls: type["Param[_T]"],
        factory: Callable[[], _T],
    ) -> Callable[[], "Param[_T]"]:
        return cast(Callable[[], Param[_T]], factory)

    def __init__(self) -> None:
        raise TypeError("Param is a marker class and cannot be instantiated directly")

    def __set__(
        self: "Param[_BaseTransformT]",
        instance: "DataModule",
        value: Optional[_BaseTransformT],
    ) -> None: ...

    @overload
    def __get__(
        self: "Param[BaseTransform[Any, Any, Any, BatchT]]",
        instance: "DataModule[AsBatch]",
        owner: type["DataModule[AsBatch]"],
    ) -> BatchT: ...

    @overload
    def __get__(
        self: "Param[BaseTransform[Any, Any, InstanceT, Any]]",
        instance: "DataModule[AsInstance]",
        owner: type["DataModule[AsInstance]"],
    ) -> InstanceT: ...

    @overload
    def __get__(
        self,
        instance: "DataModule[AsConverter]",
        owner: type["DataModule[AsConverter]"],
    ) -> _T_co: ...

    def __get__(
        self,
        instance: "DataModule",
        owner: type["DataModule"],
    ) -> Union[_T_co, Any]: ...


@dataclass_transform(kw_only_default=True, field_specifiers=(dataclasses.field,))
class BaseTransformMeta(abc.ABCMeta):
    def __new__(mcls, name, bases, namespace):
        namespace = {k: None if isinstance(v, Extra) else v for k, v in namespace.items()}
        dataclass_params = {"kw_only": True} if sys.version_info >= (3, 10) else {}
        cls = super().__new__(mcls, name, bases, namespace)
        cls = dataclasses.dataclass(**dataclass_params)(cls)
        register_dataclass(cls)
        return cls


class BaseTransform(
    Registrable,
    Generic[_S, _T, InstanceT_co, BatchT_co],
    abc.ABC,
    metaclass=BaseTransformMeta,
):
    accessor: Optional[Union[str, Callable[[_S], _T]]] = None

    _parent: Optional[type["DataModule"]] = dataclasses.field(default=None, init=False, repr=False, compare=False)
    _field_name: Optional[str] = dataclasses.field(
        default=None, init=False, repr=False, compare=False, metadata={JAX_STATIC_FIELD: True}
    )
    _training: bool = dataclasses.field(
        default=False, init=False, repr=False, compare=False, metadata={JAX_STATIC_FIELD: True}
    )
    _extra: bool = dataclasses.field(
        default=False, init=False, repr=False, compare=False, metadata={JAX_STATIC_FIELD: True}
    )

    __process_parent__: ClassVar[bool] = False

    @abc.abstractmethod
    def instance(self, obj: _T, /) -> InstanceT_co:
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def batch(self, batch: Sequence[InstanceT_co], /) -> BatchT_co:
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, data: _S, /) -> Optional[InstanceT_co]:
        value = self._get_input_value(data)
        if self._extra and value is None:
            return None
        assert value is not None
        return self.instance(value)

    @overload
    def __set__(
        self,
        instance: "DataModule[AsInstance]",
        value: Union[InstanceT_co, Self],
    ) -> None: ...

    @overload
    def __set__(
        self,
        instance: "DataModule[AsConverter]",
        value: Self,
    ) -> None: ...

    def __set__(
        self,
        instance: "DataModule",
        value: Union[InstanceT_co, Self],
    ) -> None: ...

    def __set_name__(self, owner: type["DataModule"], name: str) -> None:
        self._parent = owner
        self._field_name = name
        self._extra = name in owner.__get_extra_fields__()

    @overload
    def __get__(
        self: "BaseTransform[Any, Any, Any, BatchT_co]",
        instance: "DataModule[AsBatch]",
        owner: type["DataModule[AsBatch]"],
    ) -> BatchT_co: ...

    @overload
    def __get__(
        self: "BaseTransform[BatchT_co]",
        instance: Extra,
        owner: type[Extra],
    ) -> Optional[BatchT_co]: ...

    @overload
    def __get__(
        self,
        instance: "DataModule[AsInstance]",
        owner: type["DataModule[AsInstance]"],
    ) -> InstanceT_co: ...

    @overload
    def __get__(
        self,
        instance: Extra,
        owner: type[Extra],
    ) -> Optional[InstanceT_co]: ...

    @overload
    def __get__(
        self,
        instance: "DataModule[AsConverter]",
        owner: type["DataModule[AsConverter]"],
    ) -> Self: ...

    @overload
    def __get__(
        self,
        instance: Any,
        owner: type[Any],
    ) -> Any: ...

    def __get__(
        self,
        instance: Union["DataModule", Extra],
        owner: Union[type["DataModule"], type[Extra]],
    ) -> Union[InstanceT_co, Self, Any, Optional[BatchT_co]]: ...

    @contextmanager
    def train(self) -> Iterator[None]:
        original = self._training
        self._training = True
        try:
            if not original:
                self._on_start_training()
            yield
            if not original:
                self._on_end_training()
        finally:
            self._training = original

    def save(self, directory: Union[str, PathLike]) -> None:
        filepath = Path(directory) / "transform.pkl"
        with filepath.open("wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, directory: Union[str, PathLike]) -> Self:
        filepath = Path(directory) / "transform.pkl"
        with filepath.open("rb") as f:
            obj = cloudpickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return obj

    def _get_input_value(self, data: _S) -> Optional[_T]:
        if self._parent and self._parent.__process_parent__:
            return cast(_T, data)
        else:
            if self.accessor is None:
                if self._field_name is None:
                    raise RuntimeError("Accessor function is not set")
                accessor = partial(xgetattr, name=self._field_name)
            elif isinstance(self.accessor, str):
                accessor = partial(xgetattr, name=self.accessor)
            else:
                accessor = self.accessor
            try:
                return accessor(data)
            except (AttributeError, KeyError):
                if not self._extra:
                    raise
                return None

    def _on_start_training(self) -> None:
        pass

    def _on_end_training(self) -> None:
        pass


#
# DataModule
#

_InstanceT = TypeVar("_InstanceT", bound="DataModule[AsInstance]", default=Any)
_BatchT = TypeVar("_BatchT", bound="DataModule[AsBatch]", default=Any)
_DataModuleModeT_co = TypeVar("_DataModuleModeT_co", bound=DataModuleMode, covariant=True)


@register_dataclass
@dataclasses.dataclass
class _Unavailable: ...


_UNAVAILABLE = _Unavailable()


class DataModule(
    BaseTransform[_T, _T, _InstanceT, _BatchT],
    Generic[_DataModuleModeT_co, _T, _InstanceT, _BatchT],
):
    __is_datamodule__: ClassVar[Literal[True]] = True
    __param_fields__: ClassVar[Optional[Mapping[str, dataclasses.Field]]] = None
    __extra_fields__: ClassVar[Optional[Mapping[str, dataclasses.Field]]] = None

    _batch_size: Optional[int] = dataclasses.field(
        default=None, init=False, repr=False, compare=False, metadata={JAX_STATIC_FIELD: True}
    )
    __mode__: Optional[_DataModuleModeT_co] = dataclasses.field(
        default=None, init=False, repr=False, compare=False, metadata={JAX_STATIC_FIELD: True}
    )

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
