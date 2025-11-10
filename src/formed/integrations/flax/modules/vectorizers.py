"""Sequence vectorization modules for Flax models.

This module provides vectorizers that convert variable-length sequences
into fixed-size vectors. Vectorizers apply pooling operations over the
sequence dimension to produce single vectors per sequence.

Key Components:
    - BaseSequenceVectorizer: Abstract base class for vectorizers
    - BagOfEmbeddingsSequenceVectorizer: Pools sequence embeddings

Features:
    - Multiple pooling strategies (mean, max, min, sum, first, last, hier)
    - Masked pooling to ignore padding tokens
    - Optional normalization before pooling
    - Hierarchical pooling with sliding windows

Example:
    >>> from formed.integrations.flax.modules import BagOfEmbeddingsSequenceVectorizer
    >>>
    >>> # Mean pooling over sequence
    >>> vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="mean")
    >>> vector = vectorizer(embeddings, mask=mask)
    >>>
    >>> # Max pooling with normalization
    >>> vectorizer = BagOfEmbeddingsSequenceVectorizer(
    ...     pooling="max",
    ...     normalize=True
    ... )

"""

import abc
from typing import Literal, Optional

import jax
from colt import Registrable
from flax import nnx

from formed.integrations.flax.utils import masked_pool


class BaseSequenceVectorizer(nnx.Module, Registrable, abc.ABC):
    """Abstract base class for sequence vectorizers.

    Vectorizers convert variable-length sequences into fixed-size vectors
    by applying pooling operations over the sequence dimension.

    """

    @abc.abstractmethod
    def __call__(
        self,
        inputs: jax.Array,
        *,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Vectorize a sequence into a fixed-size vector.

        Args:
            inputs: Input embeddings of shape (batch_size, seq_len, embedding_dim).
            mask: Optional attention mask of shape (batch_size, seq_len).

        Returns:
            Vectorized output of shape (batch_size, output_dim).

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_input_dim(self) -> Optional[int]:
        """Get the expected input dimension.

        Returns:
            Input dimension or None if dimension-agnostic.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_output_dim(self) -> Optional[int]:
        """Get the output dimension.

        Returns:
            Output dimension or None if same as input.

        """
        raise NotImplementedError


@BaseSequenceVectorizer.register("boe")
@BaseSequenceVectorizer.register("bag_of_embeddings")
class BagOfEmbeddingsSequenceVectorizer(BaseSequenceVectorizer):
    """Bag-of-embeddings vectorizer using pooling operations.

    This vectorizer applies pooling over the sequence dimension to create
    fixed-size vectors. Multiple pooling strategies are supported, and
    padding tokens are properly masked during pooling.

    Args:
        pooling: Pooling strategy to use:
            - "mean": Average pooling (default)
            - "max": Max pooling
            - "min": Min pooling
            - "sum": Sum pooling
            - "first": Take first token
            - "last": Take last non-padding token
            - "hier": Hierarchical pooling with sliding window
        normalize: Whether to L2-normalize embeddings before pooling.
        window_size: Window size for hierarchical pooling (required if pooling="hier").

    Example:
        >>> # Mean pooling
        >>> vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="mean")
        >>> vector = vectorizer(embeddings, mask=mask)
        >>>
        >>> # Max pooling with normalization
        >>> vectorizer = BagOfEmbeddingsSequenceVectorizer(
        ...     pooling="max",
        ...     normalize=True
        ... )
        >>>
        >>> # Hierarchical pooling
        >>> vectorizer = BagOfEmbeddingsSequenceVectorizer(
        ...     pooling="hier",
        ...     window_size=3
        ... )

    Note:
        This vectorizer is dimension-agnostic - it preserves the embedding
        dimension from input to output.

    """

    def __init__(
        self,
        pooling: Literal["mean", "max", "min", "sum", "hier", "first", "last"] = "mean",
        normalize: bool = False,
        window_size: Optional[int] = None,
    ) -> None:
        self._pooling: Literal["mean", "max", "min", "sum", "hier", "first", "last"] = pooling
        self._normalize = normalize
        self._window_size = window_size

    def __call__(
        self,
        inputs: jax.Array,
        *,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        return masked_pool(
            inputs,
            mask=mask,
            pooling=self._pooling,
            normalize=self._normalize,
            window_size=self._window_size,
        )

    def get_input_dim(self) -> None:
        return None

    def get_output_dim(self) -> None:
        return None
