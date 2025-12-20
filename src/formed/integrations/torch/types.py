"""Type definitions and protocols for PyTorch integration.

This module defines type aliases, protocols, and generic type variables
used throughout the PyTorch integration, providing type-safe interfaces for
data loaders, evaluators, optimizers, and model components.

Key Protocols:
    - IOptimizer: Protocol for torch-compatible optimizers
    - ILRScheduler: Protocol for learning rate schedulers
    - IGradScaler: Protocol for gradient scalers (mixed precision training)
    - IDataLoader: Protocol for data loading functions
    - IEvaluator: Protocol for metric computation
    - IIDSequenceBatch: Protocol for tokenized sequence batches
    - IAnalyzedTextBatch: Protocol for analyzed text with multiple features

Type Variables:
    - ModelInputT/ModelOutputT: Model input/output types
    - TensorCompatible: Union of numpy.ndarray and torch.Tensor
    - ItemT: Generic dataset item type

"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Optional, Protocol, Union, runtime_checkable

import numpy
import torch
from colt import Lazy
from typing_extensions import TypeAlias, TypeVar


@runtime_checkable
class IOptimizer(Protocol):
    """Protocol for torch-compatible optimizer objects.

    Attributes:
        zero_grad: Function to zero out gradients.
        step: Function to update parameters.
        state_dict: Function to get optimizer state.
        load_state_dict: Function to load optimizer state.

    Note:
        This protocol does not include `param_groups` to maintain compatibility
        with Colt's isinstance checks during object construction. Access to
        param_groups should be done by casting to the concrete optimizer type.

    """

    def zero_grad(self) -> None: ...
    def step(self) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


@runtime_checkable
class ILRScheduler(Protocol):
    """Protocol for torch-compatible learning rate scheduler objects.

    Attributes:
        step: Function to update learning rates.
        state_dict: Function to get scheduler state.
        load_state_dict: Function to load scheduler state.

    """

    def step(self) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


@runtime_checkable
class IGradScaler(Protocol):
    """Protocol for gradient scaler objects used in mixed precision training.

    This protocol supports both torch.cuda.amp.GradScaler and torch.cpu.amp.GradScaler.

    Attributes:
        scale: Function to scale loss before backward pass.
        unscale_: Function to unscale gradients before gradient clipping.
        step: Function to perform optimizer step with unscaling.
        update: Function to update the scale factor.
        state_dict: Function to get scaler state.
        load_state_dict: Function to load scaler state.

    """

    def scale(self, outputs: Any) -> Any: ...
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...
    def step(self, optimizer: torch.optim.Optimizer, *args: Any, **kwargs: Any) -> Optional[float]: ...
    def update(self, new_scale: Optional[Any] = None) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


TensorCompatible: TypeAlias = Union[numpy.ndarray, torch.Tensor, Sequence[float]]
TensorCompatibleT = TypeVar("TensorCompatibleT", bound=TensorCompatible)
ItemT = TypeVar("ItemT", default=Any)
ItemT_contra = TypeVar("ItemT_contra", contravariant=True, default=Any)
ModelInputT = TypeVar("ModelInputT", default=Any)
ModelInputT_co = TypeVar("ModelInputT_co", covariant=True, default=Any)
ModelInputT_contra = TypeVar("ModelInputT_contra", contravariant=True, default=Any)
ModelOutputT = TypeVar("ModelOutputT", default=Any)
ModelOutputT_contra = TypeVar("ModelOutputT_contra", contravariant=True, default=Any)
ModelParamsT = TypeVar("ModelParamsT", default=None)
OptimizerT = TypeVar("OptimizerT", bound=IOptimizer)
OptimizerFactory = Union[Lazy[IOptimizer], IOptimizer, Callable[[Iterator[torch.nn.Parameter]], IOptimizer]]
LRSchedulerT = TypeVar("LRSchedulerT", bound=ILRScheduler)
LRSchedulerFactory: TypeAlias = Union[Lazy[ILRScheduler], ILRScheduler, Callable[[IOptimizer], ILRScheduler]]


@runtime_checkable
class IBatchIterator(Protocol[ModelInputT_co]):
    """Protocol for batch iterators.

    Type Parameters:
        ModelInputT_co: Covariant type of model input batches.

    """

    def __iter__(self) -> Iterator[ModelInputT_co]: ...
    def __len__(self) -> int: ...


@runtime_checkable
class IDataLoader(Protocol[ItemT_contra, ModelInputT_co]):
    """Protocol for data loader functions.

    A data loader transforms a sequence of items into batched model inputs.

    Type Parameters:
        ItemT_contra: Contravariant type of dataset items.
        ModelInputT_co: Covariant type of model input batches.

    """

    def __call__(self, data: Sequence[ItemT_contra]) -> IBatchIterator[ModelInputT_co]: ...


@runtime_checkable
class IStreamingDataLoader(Protocol[ItemT_contra, ModelInputT_co]):
    """Protocol for streaming data loader functions.

    A streaming data loader transforms an iterable of items into an iterable of batched model inputs.

    Type Parameters:
        ItemT_contra: Contravariant type of dataset items.
        ModelInputT_co: Covariant type of model input batches.

    """

    def __call__(self, data: Iterable[ItemT_contra]) -> Iterable[ModelInputT_co]: ...


@runtime_checkable
class IEvaluator(Protocol[ModelInputT_contra, ModelOutputT_contra]):
    """Protocol for metric evaluators.

    Evaluators accumulate predictions and compute metrics over batches.

    Type Parameters:
        ModelInputT_contra: Contravariant type of model inputs.
        ModelOutputT_contra: Contravariant type of model outputs.

    """

    def update(self, inputs: ModelInputT_contra, output: ModelOutputT_contra, /) -> None:
        """Update internal state with a batch of predictions.

        Args:
            inputs: Model inputs for the batch.
            output: Model predictions for the batch.

        """
        ...

    def compute(self) -> dict[str, float]:
        """Compute metrics from accumulated predictions.

        Returns:
            Dictionary mapping metric names to values.

        """
        ...

    def reset(self) -> None:
        """Reset internal state for new evaluation."""
        ...


@runtime_checkable
class IIDSequenceBatch(Protocol[TensorCompatibleT]):
    """Protocol for tokenized sequence batches.

    Represents batched sequences with token IDs and attention masks.

    Type Parameters:
        TensorCompatibleT: Type of array (numpy.ndarray or torch.Tensor).

    Attributes:
        ids: Token IDs of shape (batch_size, seq_len) or (batch_size, seq_len, char_len).
        mask: Attention mask of same shape as ids.

    """

    ids: TensorCompatibleT
    mask: TensorCompatibleT

    def __len__(self) -> int: ...


IDSequenceBatchT = TypeVar("IDSequenceBatchT", bound=IIDSequenceBatch)


SurfaceBatchT = TypeVar("SurfaceBatchT", bound=IIDSequenceBatch, default=Any)
PostagBatchT = TypeVar("PostagBatchT", bound=Optional[IIDSequenceBatch], default=Any)
CharacterBatchT = TypeVar("CharacterBatchT", bound=Optional[IIDSequenceBatch], default=Any)


class IAnalyzedTextBatch(Protocol[SurfaceBatchT, PostagBatchT, CharacterBatchT]):
    """Protocol for analyzed text batches with multiple linguistic features.

    Represents text with separate tokenizations for surface forms,
    part-of-speech tags, and character sequences.

    Type Parameters:
        SurfaceBatchT: Type of surface form batch.
        PostagBatchT: Type of POS tag batch (may be Optional).
        CharacterBatchT: Type of character batch (may be Optional).

    Attributes:
        surfaces: Surface form token sequences.
        postags: Part-of-speech tag sequences (optional).
        characters: Character sequences (optional).

    """

    surfaces: SurfaceBatchT
    postags: PostagBatchT
    characters: CharacterBatchT

    def __len__(self) -> int: ...


AnalyzedTextBatchT = TypeVar("AnalyzedTextBatchT", bound=IAnalyzedTextBatch, default=Any)
