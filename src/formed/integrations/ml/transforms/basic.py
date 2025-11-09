"""Basic data transformations for common machine learning tasks.

This module provides fundamental transform classes for handling common data types
in machine learning pipelines, including labels, scalars, tensors, and metadata.

Available Transforms:
    - MetadataTransform: Pass-through transform for metadata (e.g., IDs, names)
    - LabelIndexer: Map labels to integer indices with vocabulary building
    - ScalarTransform: Convert scalar values to numpy arrays
    - TensorTransform: Convert numpy arrays to batched tensors

Example:
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

from ..types import LabelT
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

    Example:
        >>> transform = MetadataTransform(accessor="id")
        >>> instance = transform({"id": "example_001"})  # Returns "example_001"
        >>> batch = transform.batch(["example_001", "example_002", "example_003"])
        >>> print(batch)  # ["example_001", "example_002", "example_003"]

    Note:
        This transform is stateless and does not require training.
    """

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

    Example:
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

        Example:
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

    Example:
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

    Example:
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
