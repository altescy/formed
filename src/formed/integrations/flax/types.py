"""Type definitions and protocols for Flax integration.

This module defines type aliases, protocols, and generic type variables
used throughout the Flax integration, providing type-safe interfaces for
data loaders, evaluators, optimizers, and model components.

Key Protocols:
    - IOptimizer: Protocol for optax-compatible optimizers
    - IDataLoader: Protocol for data loading functions
    - IEvaluator: Protocol for metric computation
    - IIDSequenceBatch: Protocol for tokenized sequence batches
    - IAnalyzedTextBatch: Protocol for analyzed text with multiple features

Type Variables:
    - ModelInputT/ModelOutputT: Model input/output types
    - ArrayCompatible: Union of numpy.ndarray and jax.Array
    - ItemT: Generic dataset item type

"""

from collections.abc import Iterator, Sequence
from typing import Any, Optional, Protocol, Union, runtime_checkable

import jax
import numpy
import optax
from typing_extensions import TypeAlias, TypeVar


@runtime_checkable
class IOptimizer(Protocol):
    """Protocol for optax-compatible optimizer objects.

    Attributes:
        init: Function to initialize optimizer state.
        update: Function to update parameters with gradients.

    """

    init: optax.TransformInitFn
    update: optax.TransformUpdateFn


ArrayCompatible: TypeAlias = Union[numpy.ndarray, jax.Array, Sequence[float]]
ArrayCompatibleT = TypeVar("ArrayCompatibleT", bound=ArrayCompatible)
ItemT = TypeVar("ItemT", default=Any)
ItemT_contra = TypeVar("ItemT_contra", contravariant=True, default=Any)
ModelInputT = TypeVar("ModelInputT", default=Any)
ModelInputT_co = TypeVar("ModelInputT_co", covariant=True, default=Any)
ModelInputT_contra = TypeVar("ModelInputT_contra", contravariant=True, default=Any)
ModelOutputT = TypeVar("ModelOutputT", default=Any)
ModelOutputT_contra = TypeVar("ModelOutputT_contra", contravariant=True, default=Any)
ModelParamsT = TypeVar("ModelParamsT", default=None)
OptimizerT = TypeVar("OptimizerT", bound=IOptimizer)


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
class IIDSequenceBatch(Protocol[ArrayCompatibleT]):
    """Protocol for tokenized sequence batches.

    Represents batched sequences with token IDs and attention masks.

    Type Parameters:
        ArrayCompatibleT: Type of array (numpy.ndarray or jax.Array).

    Attributes:
        ids: Token IDs of shape (batch_size, seq_len) or (batch_size, seq_len, char_len).
        mask: Attention mask of same shape as ids.

    """

    ids: ArrayCompatibleT
    mask: ArrayCompatibleT

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
