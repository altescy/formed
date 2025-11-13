"""Sequence vectorization modules for PyTorch models.

This module provides vectorizers that convert variable-length sequences
into fixed-size vectors. Vectorizers apply pooling operations over the
sequence dimension to produce single vectors per sequence.

Key Components:
    - BaseSequenceVectorizer: Abstract base class for vectorizers
    - BagOfEmbeddingsSequenceVectorizer: Pools sequence embeddings

Features:
    - Multiple pooling strategies (mean, max, min, sum, first, last)
    - Masked pooling to ignore padding tokens
    - Optional normalization before pooling

Example:
    >>> from formed.integrations.torch.modules import BagOfEmbeddingsSequenceVectorizer
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
from collections.abc import Callable, Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from colt import Registrable

PoolingMethod = Literal["mean", "max", "min", "sum", "first", "last"]


def masked_pool(
    inputs: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    pooling: PoolingMethod | Sequence[PoolingMethod] = "mean",
    normalize: bool = False,
) -> torch.Tensor:
    """Apply masked pooling over the sequence dimension.

    Args:
        inputs: Input tensor of shape (batch_size, seq_len, feature_dim).
        mask: Mask tensor of shape (batch_size, seq_len). True/1 indicates valid positions.
        pooling: Pooling method or sequence of methods.
        normalize: Whether to L2-normalize before pooling.

    Returns:
        Pooled tensor of shape (batch_size, feature_dim * num_pooling_methods).

    """
    if normalize:
        inputs = F.normalize(inputs, p=2, dim=-1)

    if mask is None:
        mask = torch.ones(inputs.shape[:-1], dtype=torch.bool, device=inputs.device)

    # Convert mask to boolean if needed
    if mask.dtype != torch.bool:
        mask = mask.bool()

    pooling_methods = [pooling] if isinstance(pooling, str) else list(pooling)
    results = []

    for method in pooling_methods:
        if method == "mean":
            # Masked mean
            masked_inputs = inputs * mask.unsqueeze(-1)
            pooled = masked_inputs.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        elif method == "max":
            # Masked max
            masked_inputs = inputs.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled, _ = masked_inputs.max(dim=1)
        elif method == "min":
            # Masked min
            masked_inputs = inputs.masked_fill(~mask.unsqueeze(-1), float("inf"))
            pooled, _ = masked_inputs.min(dim=1)
        elif method == "sum":
            # Masked sum
            masked_inputs = inputs * mask.unsqueeze(-1)
            pooled = masked_inputs.sum(dim=1)
        elif method == "first":
            # First token
            pooled = inputs[:, 0, :]
        elif method == "last":
            # Last valid token
            # Find the index of the last valid token for each sequence
            lengths = mask.sum(dim=1).clamp(min=1) - 1  # -1 because indices are 0-based
            batch_indices = torch.arange(inputs.size(0), device=inputs.device)
            pooled = inputs[batch_indices, lengths.long()]
        else:
            raise ValueError(f"Unknown pooling method: {method}")

        results.append(pooled)

    return torch.cat(results, dim=-1) if len(results) > 1 else results[0]


class BaseSequenceVectorizer(nn.Module, Registrable, abc.ABC):
    """Abstract base class for sequence vectorizers.

    Vectorizers convert variable-length sequences into fixed-size vectors
    by applying pooling operations over the sequence dimension.

    """

    @abc.abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vectorize a sequence into a fixed-size vector.

        Args:
            inputs: Input embeddings of shape (batch_size, seq_len, embedding_dim).
            mask: Optional attention mask of shape (batch_size, seq_len).

        Returns:
            Vectorized output of shape (batch_size, output_dim).

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_input_dim(self) -> int | None:
        """Get the expected input dimension.

        Returns:
            Input dimension or None if dimension-agnostic.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_output_dim(self) -> int | Callable[[int], int]:
        """Get the output dimension.

        Returns:
            Output feature dimension or a function mapping input dim to output dim.

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
        normalize: Whether to L2-normalize embeddings before pooling.

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
        >>> # Multiple pooling methods combined
        >>> vectorizer = BagOfEmbeddingsSequenceVectorizer(
        ...     pooling=["mean", "max"]
        ... )

    Note:
        This vectorizer is dimension-agnostic - it preserves the embedding
        dimension from input to output (multiplied by number of pooling methods).

    """

    def __init__(
        self,
        pooling: PoolingMethod | Sequence[PoolingMethod] = "mean",
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self._pooling: PoolingMethod | Sequence[PoolingMethod] = pooling
        self._normalize = normalize

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return masked_pool(
            inputs,
            mask=mask,
            pooling=self._pooling,
            normalize=self._normalize,
        )

    def get_input_dim(self) -> None:
        return None

    def get_output_dim(self) -> Callable[[int], int]:
        num_pooling = 1 if isinstance(self._pooling, str) else len(self._pooling)
        return lambda input_dim: input_dim * num_pooling
