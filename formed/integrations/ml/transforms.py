import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from os import PathLike
from typing import Any, Generic, Optional, TypeVar, Union, cast

import dill
from colt import Registrable

from .fields import Field, LabelField, ListField, ScalarField, TensorField, TextField
from .indexers import LabelIndexer, TokenIndexer
from .types import DataArray, IntTensor, ScalarT, Tensor, TensorT  # noqa: F401
from .utils import RegexTokenizer

T = TypeVar("T")
FieldT = TypeVar("FieldT", bound=Field)
FieldT_contra = TypeVar("FieldT_contra", bound=Field, covariant=True)
HashableT = TypeVar("HashableT", bound=Hashable)
FieldTransformT = TypeVar("FieldTransformT", bound="FieldTransform")


class FieldTransform(Registrable, Generic[T, FieldT_contra]):
    def __call__(self, obj: T, /) -> FieldT_contra:
        raise NotImplementedError

    def reconstruct(self, array: DataArray, /) -> T:
        raise NotImplementedError

    def build(self, dataset: Iterable[T], /) -> None:
        pass

    def freeze(self) -> None:
        pass

    def stats(self) -> Mapping[str, Any]:
        return {}

    def save(self, path: Union[str, PathLike], /) -> None:
        with open(path, "wb") as file:
            dill.dump(self, file)

    @classmethod
    def load(cls: type[FieldTransformT], path: Union[str, PathLike], /) -> FieldTransformT:
        with open(path, "rb") as file:
            return cast(FieldTransformT, dill.load(file))


@FieldTransform.register("text")
class TextFieldTransform(
    Generic[HashableT],
    FieldTransform[Union[str, Sequence[HashableT]], TextField[HashableT]],
):
    def __init__(
        self,
        tokenizer: Optional[Callable[[str], Sequence[HashableT]]] = None,
        *,
        pad_token: Optional[HashableT] = None,
        unk_token: Optional[HashableT] = None,
        reserved_tokens: Sequence[HashableT] = (),
        indexer: Optional[TokenIndexer[HashableT]] = None,
    ) -> None:
        if indexer is not None and (pad_token is not None or unk_token is not None):
            warnings.warn(
                "pad_token and unk_token are ignored when indexer is provided",
                UserWarning,
            )

        self._tokenizer: Callable[[str], Sequence[HashableT]] = (
            tokenizer or RegexTokenizer()  # type: ignore[assignment]
        )
        self._indexer: TokenIndexer[HashableT] = (
            indexer if indexer is not None else TokenIndexer(default=unk_token, reserved=reserved_tokens)
        )

    @property
    def padding_value(self) -> int:
        return self._indexer[self._indexer.default] if self._indexer.default is not None else 0

    def __call__(self, obj: Union[str, Sequence[HashableT]], /) -> TextField:
        tokens = self._tokenizer(obj) if isinstance(obj, str) else obj
        return TextField(
            tokens,
            indexer=self._indexer,
            padding_value=self.padding_value,
        )

    def reconstruct(self, array: DataArray, /) -> Union[str, Sequence[HashableT]]:
        assert isinstance(array, Mapping)
        field: TextField[HashableT] = TextField.from_array(
            array,
            indexer=self._indexer,
            padding_value=self.padding_value,
        )
        return field.tokens

    def build(self, dataset: Iterable[Union[str, Sequence[HashableT]]], /) -> None:
        if self._indexer.is_frozen:
            return
        for text in dataset:
            tokens = self._tokenizer(text) if isinstance(text, str) else text
            for token in tokens:
                self._indexer.add(token)

    def freeze(self) -> None:
        self._indexer.freeze()

    def stats(self) -> Mapping[str, Any]:
        return {
            "index_size": len(self._indexer),
        }


@FieldTransform.register("label")
class LabelFieldTransform(FieldTransform[HashableT, LabelField]):
    def __init__(
        self,
        indexer: Optional[LabelIndexer[HashableT]] = None,
    ) -> None:
        self._indexer = indexer if indexer is not None else LabelIndexer[HashableT]()

    def __call__(self, obj: HashableT, /) -> LabelField:
        return LabelField(obj, indexer=self._indexer)

    def reconstruct(self, array: DataArray, /) -> HashableT:
        array = cast(IntTensor, array)
        field: LabelField[HashableT] = LabelField.from_array(array, indexer=self._indexer)
        return field.label

    def build(self, dataset: Iterable[HashableT], /) -> None:
        if self._indexer.is_frozen:
            return
        for label in dataset:
            self._indexer.add(label)

    def freeze(self) -> None:
        self._indexer.freeze()

    def stats(self) -> Mapping[str, Any]:
        return {
            "index_size": len(self._indexer),
        }


@FieldTransform.register("list")
class ListFieldTransform(
    Generic[T],
    FieldTransform[Sequence[T], ListField],
):
    def __init__(
        self,
        transform: FieldTransform[T, Field[Any]],
        padding_value: Optional[int] = None,
    ) -> None:
        self._transform = transform
        self._padding_value = padding_value

    def __call__(self, obj: Sequence[T], /) -> ListField:
        return ListField([self._transform(value) for value in obj], padding_value=self._padding_value)

    def reconstruct(self, array: DataArray, /) -> Sequence[T]:
        assert isinstance(array, Sequence)
        return [self._transform.reconstruct(value) for value in array]

    def build(self, dataset: Iterable[Sequence[T]], /) -> None:
        self._transform.build((value for values in dataset for value in values))

    def freeze(self) -> None:
        self._transform.freeze()

    def stats(self) -> Mapping[str, Any]:
        return self._transform.stats()


@FieldTransform.register("scalar")
class ScalarFieldTransform(
    FieldTransform[ScalarT, ScalarField],
):
    def __call__(self, obj: ScalarT, /) -> ScalarField:
        return ScalarField(obj)

    def reconstruct(self, array: DataArray, /) -> ScalarT:
        return cast(ScalarT, array)

    def build(self, dataset: Iterable[ScalarT], /) -> None:
        pass

    def freeze(self) -> None:
        pass


@FieldTransform.register("tensor")
class TensorFieldTransform(
    FieldTransform[TensorT, TensorField],
):
    def __call__(self, obj: TensorT, /) -> TensorField:
        return TensorField(obj)

    def reconstruct(self, array: DataArray, /) -> TensorT:
        return cast(TensorT, array)

    def build(self, dataset: Iterable[TensorT], /) -> None:
        pass

    def freeze(self) -> None:
        pass
