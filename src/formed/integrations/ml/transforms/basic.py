"""Basic data transformations for common machine learning tasks.

This module provides fundamental transform classes for handling common data types
in machine learning pipelines, including labels, scalars, tensors, and metadata.

Available Transforms:
    - MetadataTransform: Pass-through transform for metadata (e.g., IDs, names)
    - LabelIndexer: Map labels to integer indices with vocabulary building
    - ScalarTransform: Convert scalar values to numpy arrays
    - TensorTransform: Convert numpy arrays to batched tensors

Examples:
    >>> from formed.integrations.ml import LabelIndexer, ScalarTransform
    >>>
    >>> # Label indexing with vocabulary building
    >>> label_indexer = LabelIndexer()
    >>> with label_indexer.train():
    ...     idx1 = label_indexer.instance("positive")  # Returns 0
    ...     idx2 = label_indexer.instance("negative")  # Returns 1
    >>> batch = label_indexer.batch([0, 1, 0])  # np.array([0, 1, 0])
    >>>
    >>> # Scalar to tensor
    >>> scalar_transform = ScalarTransform()
    >>> values = [1.5, 2.3, 4.1]
    >>> batch = scalar_transform.batch(values)  # np.array([1.5, 2.3, 4.1])

"""

import dataclasses
import operator
from collections.abc import Sequence
from contextlib import suppress
from logging import getLogger
from typing import Any, Generic

import numpy
from typing_extensions import TypeVar

from ..types import LabelT, VariableTensorBatch
from .base import BaseTransform

logger = getLogger(__name__)


_S = TypeVar("_S", default=Any)
_T = TypeVar("_T", default=Any)


@BaseTransform.register("metadata")
class MetadataTransform(
    Generic[_S, _T],
    BaseTransform[_S, _T, _T, Sequence[_T]],
):
    """Pass-through transform for metadata fields.

    MetadataTransform does not modify data during instance transformation and
    simply collects values into a list during batching. This is useful for
    metadata like IDs, filenames, or other non-numerical information that should
    be preserved but not transformed into tensors.

    Type Parameters:
        _S: Source data type before accessor
        _T: Value type (same as instance and element of batch)

    Examples:
        >>> transform = MetadataTransform(accessor="id")
        >>> instance = transform({"id": "example_001"})  # Returns "example_001"
        >>> batch = transform.batch(["example_001", "example_002", "example_003"])
        >>> print(batch)  # ["example_001", "example_002", "example_003"]

    Note:
        This transform is stateless and does not require training.

    """

    __is_static__ = True

    def instance(self, value: _T, /) -> _T:
        return value

    def batch(self, batch: Sequence[_T], /) -> Sequence[_T]:
        return list(batch)


@BaseTransform.register("label")
class LabelIndexer(BaseTransform[_S, LabelT, int, numpy.ndarray], Generic[_S, LabelT]):
    """Map labels to integer indices with vocabulary building and statistics tracking.

    LabelIndexer maintains a bidirectional mapping between labels and integer indices.
    In training mode, it dynamically builds the label vocabulary and tracks label
    frequencies. The vocabulary can be frozen to prevent changes during inference.

    Type Parameters:
        _S: Source data type before accessor
        LabelT: Label type (must be hashable)

    Attributes:
        label2id: Pre-defined label-to-index mapping. If empty, built during training.
        freeze: If True, prevent vocabulary updates even in training mode.

    Properties:
        num_labels: Total number of unique labels in vocabulary.
        labels: List of labels sorted by their indices.
        occurrences: Dictionary mapping labels to their occurrence counts.
        distribution: Smoothed probability distribution over labels.

    Examples:
        >>> # Dynamic vocabulary building
        >>> indexer = LabelIndexer()
        >>> with indexer.train():
        ...     idx1 = indexer.instance("positive")  # 0
        ...     idx2 = indexer.instance("negative")  # 1
        ...     idx3 = indexer.instance("positive")  # 0 (already in vocab)
        >>> print(indexer.labels)  # ["positive", "negative"]
        >>> print(indexer.occurrences)  # {"positive": 2, "negative": 1}
        >>>
        >>> # Pre-defined vocabulary
        >>> indexer = LabelIndexer(label2id=[("positive", 0), ("negative", 1)])
        >>> idx = indexer.instance("positive")  # 0
        >>>
        >>> # Batching and reconstruction
        >>> batch = indexer.batch([0, 1, 0])  # np.array([0, 1, 0])
        >>> labels = indexer.reconstruct(batch)  # ["positive", "negative", "positive"]

    Note:
        - Raises KeyError if a label is not in vocabulary during inference
        - Use freeze=True to prevent accidental vocabulary updates
        - Distribution uses Laplace smoothing (add-one)

    """

    label2id: Sequence[tuple[LabelT, int]] = dataclasses.field(default_factory=list)
    freeze: bool = dataclasses.field(default=False)

    _label_counts: list[tuple[LabelT, int]] = dataclasses.field(
        default_factory=list, init=False, repr=False, compare=False
    )

    @property
    def num_labels(self) -> int:
        """Get the total number of unique labels in the vocabulary."""
        return len(self.label2id)

    @property
    def labels(self) -> list[LabelT]:
        """Get the list of labels sorted by their indices."""
        return [label for label, _ in sorted(self.label2id, key=operator.itemgetter(1))]

    @property
    def occurrences(self) -> dict[LabelT, int]:
        """Get the occurrence counts for each label seen during training."""
        return dict(self._label_counts)

    @property
    def distribution(self) -> numpy.ndarray:
        """Get the smoothed probability distribution over labels.

        Uses Laplace (add-one) smoothing to handle zero counts.

        Returns:
            Array of probabilities summing to 1.0, one per label.

        """
        total = sum(count for _, count in self._label_counts) + self.num_labels
        counts = [count for _, count in sorted(self._label_counts, key=operator.itemgetter(1))]
        return numpy.array([(count + 1) / total for count in counts], dtype=numpy.float32)

    def _on_start_training(self) -> None:
        self._label_counts.clear()

    def _on_end_training(self) -> None:
        pass

    def get_index(self, value: LabelT, /) -> int:
        """Get the integer index for a label.

        Args:
            value: The label to look up.

        Returns:
            The integer index associated with the label.

        Raises:
            KeyError: If the label is not in the vocabulary.

        """
        with suppress(StopIteration):
            return next(label_id for label, label_id in self.label2id if label == value)
        raise KeyError(value)

    def get_value(self, index: int, /) -> LabelT:
        """Get the label for an integer index.

        Args:
            index: The integer index to look up.

        Returns:
            The label associated with the index.

        Raises:
            KeyError: If the index is not in the vocabulary.

        """
        for label, label_id in self.label2id:
            if label_id == index:
                return label
        raise KeyError(index)

    def ingest(self, value: LabelT, /) -> None:
        """Add a label to the vocabulary and update statistics.

        This method is called internally during training to build the vocabulary
        and track label frequencies.

        Args:
            value: The label to ingest.

        Note:
            Only effective when in training mode and freeze=False.
            Logs a warning if called outside training mode.

        """
        if self.freeze:
            return
        if self._training:
            try:
                self.get_index(value)
            except KeyError:
                self.label2id = list(self.label2id) + [(value, len(self.label2id))]
                self._label_counts.append((value, 0))
            for index, (label, count) in enumerate(self._label_counts):
                if label == value:
                    self._label_counts[index] = (label, count + 1)
                    break
        else:
            logger.warning("Ignoring ingest call when not in training mode")

    def instance(self, label: LabelT, /) -> int:
        if self._training:
            self.ingest(label)
        return self.get_index(label)

    def batch(self, batch: Sequence[int], /) -> numpy.ndarray:
        return numpy.array(batch, dtype=numpy.int64)

    def reconstruct(self, batch: numpy.ndarray, /) -> list[LabelT]:
        """Convert a batch of indices back to labels.

        Args:
            batch: Array of integer indices.

        Returns:
            List of labels corresponding to the indices.

        Examples:
            >>> indexer = LabelIndexer(label2id=[("cat", 0), ("dog", 1)])
            >>> indices = numpy.array([0, 1, 0])
            >>> labels = indexer.reconstruct(indices)
            >>> print(labels)  # ["cat", "dog", "cat"]

        """
        return [self.get_value(index) for index in batch.tolist()]


@BaseTransform.register("scalar")
class ScalarTransform(
    Generic[_S],
    BaseTransform[_S, float, float, numpy.ndarray],
):
    """Transform scalar values into batched numpy arrays.

    ScalarTransform is a simple pass-through transform that preserves scalar
    values during instance transformation and stacks them into a 1D numpy array
    during batching.

    Type Parameters:
        _S: Source data type before accessor

    Examples:
        >>> transform = ScalarTransform(accessor="score")
        >>> value = transform({"score": 0.85})  # Returns 0.85
        >>> batch = transform.batch([0.85, 0.92, 0.78])
        >>> print(batch)  # np.array([0.85, 0.92, 0.78], dtype=float32)
        >>> print(batch.shape)  # (3,)

    Note:
        - Instance values remain as Python floats
        - Batch values are converted to float32 numpy arrays
        - Stateless transform, no training required

    """

    def instance(self, value: float, /) -> float:
        return value

    def batch(self, batch: Sequence[float], /) -> numpy.ndarray:
        return numpy.array(batch, dtype=numpy.float32)


@BaseTransform.register("tensor")
class TensorTransform(
    Generic[_S],
    BaseTransform[_S, numpy.ndarray, numpy.ndarray, numpy.ndarray],
):
    """Transform numpy arrays into batched tensors.

    TensorTransform preserves numpy arrays during instance transformation and
    stacks them along the batch dimension (axis 0) during batching. All arrays
    in a batch must have the same shape.

    Type Parameters:
        _S: Source data type before accessor

    Examples:
        >>> import numpy as np
        >>> transform = TensorTransform(accessor="features")
        >>> arr = transform({"features": np.array([1.0, 2.0, 3.0])})
        >>> print(arr)  # np.array([1.0, 2.0, 3.0])
        >>>
        >>> batch = transform.batch([
        ...     np.array([1.0, 2.0, 3.0]),
        ...     np.array([4.0, 5.0, 6.0]),
        ... ])
        >>> print(batch.shape)  # (2, 3)

    Note:
        - Requires all arrays in a batch to have compatible shapes
        - Stacks along axis 0 (batch dimension)
        - Stateless transform, no training required

    Raises:
        ValueError: If arrays have incompatible shapes for stacking.

    """

    def instance(self, value: numpy.ndarray, /) -> numpy.ndarray:
        return value

    def batch(self, batch: Sequence[numpy.ndarray], /) -> numpy.ndarray:
        return numpy.stack(batch, axis=0)


@BaseTransform.register("variable_tensor")
class VariableTensorTransform(
    Generic[_S],
    BaseTransform[_S, numpy.ndarray, numpy.ndarray, VariableTensorBatch[numpy.ndarray]],
):
    """Transform variable-size numpy arrays into padded batched tensors.

    VariableTensorTransform preserves numpy arrays during instance transformation
    and pads them to the maximum shape in the batch during batching. It returns
    a TensorBatch containing the padded tensor and a mask indicating valid data.

    Type Parameters:
        _S: Source data type before accessor

    Examples:
        >>> import numpy as np
        >>> transform = VariableTensorTransform(accessor="sequences")
        >>> arr = transform({"sequences": np.array([1, 2, 3])})
        >>> print(arr)  # np.array([1, 2, 3])
        >>>
        >>> batch = transform.batch([
        ...     np.array([1, 2, 3]),
        ...     np.array([4, 5]),
        ...     np.array([6, 7, 8, 9]),
        ... ])
        >>> print(batch.tensor)
        >>> # np.array([
        >>> #   [1, 2, 3, 0],
        >>> #   [4, 5, 0, 0],
        >>> #   [6, 7, 8, 9]
        >>> # ])
        >>> print(batch.mask)
        >>> # np.array([
        >>> #   [True, True, True, False],
        >>> #   [True, True, False, False],
        >>> #   [True, True, True, True]
        >>> # ])

    Note:
        - Pads arrays with zeros to match the maximum shape in the batch
        - Mask indicates which elements are valid (True) vs. padded (False)
        - Stateless transform, no training required

    """

    def instance(self, value: numpy.ndarray, /) -> numpy.ndarray:
        return value

    def batch(self, batch: Sequence[numpy.ndarray], /) -> VariableTensorBatch[numpy.ndarray]:
        if len(batch) == 0:
            return VariableTensorBatch(
                tensor=numpy.array([], dtype=numpy.float32),
                mask=numpy.array([], dtype=numpy.bool_),
            )
        max_ndim = max(arr.ndim for arr in batch)
        max_shape = []
        for dim in range(max_ndim):
            dim_sizes = [arr.shape[dim] if dim < arr.ndim else 1 for arr in batch]
            max_shape.append(max(dim_sizes))

        tensor = numpy.zeros((len(batch), *max_shape), dtype=batch[0].dtype)
        mask = numpy.zeros((len(batch), *max_shape), dtype=numpy.bool_)

        for i, arr in enumerate(batch):
            slices = tuple(slice(0, dim_size) for dim_size in arr.shape)
            tensor[i][slices] = arr
            mask[i][slices] = True

        return VariableTensorBatch[numpy.ndarray](tensor=tensor, mask=mask)


@BaseTransform.register("tensor_sequence")
class TensorSequenceTransform(
    Generic[_S],
    BaseTransform[_S, Sequence[numpy.ndarray], numpy.ndarray, VariableTensorBatch[numpy.ndarray]],
):
    """Transform sequences of numpy arrays into padded batched tensors.

    TensorSequenceTransform handles sequences of arrays (e.g., token-level embeddings)
    by stacking them into a 2D array during instance transformation, then padding
    across the batch dimension during batching.

    This is useful for token-level features where each token has its own vector,
    and different instances may have different numbers of tokens.

    Type Parameters:
        _S: Source data type before accessor

    Examples:
        >>> import numpy as np
        >>> transform = TensorSequenceTransform(accessor="token_vectors")
        >>>
        >>> # Each instance has a sequence of token vectors
        >>> data1 = {"token_vectors": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]}
        >>> data2 = {"token_vectors": [np.array([5.0, 6.0])]}
        >>>
        >>> instance1 = transform(data1)  # Shape: (2, 2) - 2 tokens, 2 dims
        >>> instance2 = transform(data2)  # Shape: (1, 2) - 1 token, 2 dims
        >>>
        >>> batch = transform.batch([instance1, instance2])
        >>> print(batch.tensor.shape)  # (2, 2, 2) - batch_size, max_tokens, dims
        >>> print(batch.mask.shape)    # (2, 2, 2)
        >>> print(batch.mask[0])  # [[True, True], [True, True]]
        >>> print(batch.mask[1])  # [[True, True], [False, False]]

    Note:
        - Converts sequence of arrays to 2D array during instance transformation
        - Pads to max token count during batching
        - Mask indicates valid tokens vs. padding
        - Empty sequences result in empty arrays

    """

    def instance(self, value: Sequence[numpy.ndarray], /) -> numpy.ndarray:
        if len(value) == 0:
            return numpy.array([], dtype=numpy.float32)
        # Stack sequence of arrays into 2D array (num_tokens, embedding_dim)
        return numpy.stack([numpy.asarray(arr) for arr in value], axis=0)

    def batch(self, batch: Sequence[numpy.ndarray], /) -> VariableTensorBatch[numpy.ndarray]:
        if len(batch) == 0:
            return VariableTensorBatch(
                tensor=numpy.array([], dtype=numpy.float32),
                mask=numpy.array([], dtype=numpy.bool_),
            )
        max_ndim = max(arr.ndim for arr in batch)
        max_shape = []
        for dim in range(max_ndim):
            dim_sizes = [arr.shape[dim] if dim < arr.ndim else 1 for arr in batch]
            max_shape.append(max(dim_sizes))

        tensor = numpy.zeros((len(batch), *max_shape), dtype=batch[0].dtype)
        mask = numpy.zeros((len(batch), *max_shape), dtype=numpy.bool_)

        for i, arr in enumerate(batch):
            slices = tuple(slice(0, dim_size) for dim_size in arr.shape)
            tensor[i][slices] = arr
            mask[i][slices] = True

        return VariableTensorBatch[numpy.ndarray](tensor=tensor, mask=mask)
