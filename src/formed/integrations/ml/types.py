import dataclasses
import enum
from collections.abc import Hashable, Sized
from contextlib import suppress
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from typing_extensions import TypeAlias, TypeVar

if TYPE_CHECKING:
    with suppress(ImportError):
        import numpy
    with suppress(ImportError):
        import torch

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

Label: TypeAlias = Hashable
LabelT = TypeVar("LabelT", bound=Label, default=Any)

Batch: TypeAlias = Sized
BatchT = TypeVar("BatchT", bound=Batch, default=Any)
BatchT_co = TypeVar("BatchT_co", bound=Batch, covariant=True, default=Any)
BatchT_contra = TypeVar("BatchT_contra", bound=Batch, contravariant=True)
InstanceT = TypeVar("InstanceT", default=Any)
InstanceT_co = TypeVar("InstanceT_co", covariant=True, default=Any)
InstanceT_contra = TypeVar("InstanceT_contra", contravariant=True)


AnyTensor: TypeAlias = Union["numpy.ndarray", "torch.Tensor"]
TensorT = TypeVar("TensorT", bound=AnyTensor, default="numpy.ndarray")
TensorT_co = TypeVar("TensorT_co", bound=AnyTensor, covariant=True, default="numpy.ndarray")
CollateContextT = TypeVar("CollateContextT")


class DataModuleMode(enum.Enum):
    AS_BATCH = enum.auto()
    AS_INSTANCE = enum.auto()
    AS_CONVERTER = enum.auto()


AsBatch: TypeAlias = Literal[DataModuleMode.AS_BATCH]
AsInstance: TypeAlias = Literal[DataModuleMode.AS_INSTANCE]
AsConverter: TypeAlias = Literal[DataModuleMode.AS_CONVERTER]

DataModuleModeT = TypeVar("DataModuleModeT", bound=DataModuleMode, default="AsConverter")


@runtime_checkable
class SupportsReconstruct(Protocol[T_co, BatchT_contra]):
    def reconstruct(self, batch: BatchT_contra, /) -> Sequence[T_co]: ...


@dataclasses.dataclass
class IDSequenceBatch(Generic[TensorT_co]):
    ids: TensorT_co
    mask: TensorT_co

    def __len__(self) -> int:
        return len(self.ids)


@runtime_checkable
class IIDSequenceBatch(Protocol[TensorT]):
    ids: TensorT
    mask: TensorT

    def __len__(self) -> int: ...


class AnalyzedText(NamedTuple):
    surfaces: Sequence[str]
    postags: Optional[Sequence[str]] = None
