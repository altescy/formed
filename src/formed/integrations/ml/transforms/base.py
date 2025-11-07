import abc
import dataclasses
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Optional, Union, cast, overload

import cloudpickle
from colt import Registrable
from typing_extensions import Self, TypeVar, dataclass_transform

from formed.common.attributeutils import xgetattr

from ..types import (
    AsBatch,
    AsConverter,
    AsInstance,
    BatchT,
    BatchT_co,
    InstanceT,
    InstanceT_co,
)

if TYPE_CHECKING:
    from .datamodule import DataModule


logger = getLogger(__name__)


_S = TypeVar("_S", default=Any)
_T = TypeVar("_T", default=Any)
_T_co = TypeVar("_T_co", covariant=True)
_BaseTransformT = TypeVar(
    "_BaseTransformT",
    bound="BaseTransform",
)
_BaseTransformT_co = TypeVar(
    "_BaseTransformT_co",
    bound="BaseTransform",
    covariant=True,
)


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
        return cls


class BaseTransform(
    Registrable,
    Generic[_S, _T, InstanceT_co, BatchT_co],
    abc.ABC,
    metaclass=BaseTransformMeta,
):
    accessor: Optional[Union[str, Callable[[_S], _T]]] = None

    _parent: Optional[type["DataModule"]] = dataclasses.field(default=None, init=False, repr=False, compare=False)
    _field_name: Optional[str] = dataclasses.field(default=None, init=False, repr=False, compare=False)
    _training: bool = dataclasses.field(default=False, init=False, repr=False, compare=False)
    _extra: bool = dataclasses.field(default=False, init=False, repr=False, compare=False)

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
