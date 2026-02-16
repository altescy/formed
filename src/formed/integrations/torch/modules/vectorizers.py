"""Sequence vectorization modules for PyTorch models.

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
from typing import Optional, Union, cast

import torch
from colt import Registrable

from ..utils import PoolingMethod, masked_pool, masked_softmax, min_value_of_dtype
from .feedforward import FeedForward


class BaseSequenceVectorizer(torch.nn.Module, Registrable, abc.ABC):
    """Abstract base class for sequence vectorizers.

    Vectorizers convert variable-length sequences into fixed-size vectors
    by applying pooling operations over the sequence dimension.

    """

    @abc.abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
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
    def get_input_dim(self) -> Optional[int]:
        """Get the expected input dimension.

        Returns:
            Input dimension or None if dimension-agnostic.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_output_dim(self) -> Union[int, Callable[[int], int]]:
        """Get the output dimension.

        Returns:
            Output feature dimension or a function mapping input dim to output dim.

        """
        raise NotImplementedError

    def __call__(
        self,
        inputs: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().__call__(inputs, mask=mask)


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
        >>> # Multiple pooling methods combined
        >>> vectorizer = BagOfEmbeddingsSequenceVectorizer(
        ...     pooling=["mean", "max"]
        ... )
        >>>
        >>> # Hierarchical pooling
        >>> vectorizer = BagOfEmbeddingsSequenceVectorizer(
        ...     pooling="hier",
        ...     window_size=3
        ... )

    Note:
        This vectorizer is dimension-agnostic - it preserves the embedding
        dimension from input to output (multiplied by number of pooling methods).

    """

    def __init__(
        self,
        pooling: Union[PoolingMethod, Sequence[PoolingMethod]] = "mean",
        normalize: bool = False,
        window_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._pooling: Union[PoolingMethod, Sequence[PoolingMethod]] = pooling
        self._normalize = normalize
        self._window_size = window_size

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Vectorize sequence using bag-of-embeddings pooling.

        Args:
            inputs: Input embeddings of shape (batch_size, seq_len, input_dim).
            mask: Optional attention mask of shape (batch_size, seq_len).
                 True indicates valid positions, False indicates padding.

        Returns:
            Vectorized output of shape (batch_size, output_dim).
            If multiple pooling methods are used, output_dim = input_dim * num_pooling.

        """
        return masked_pool(
            inputs,
            mask=mask,
            pooling=self._pooling,
            normalize=self._normalize,
            window_size=self._window_size,
        )

    def get_input_dim(self) -> None:
        """Get the expected input dimension.

        Returns:
            None (dimension-agnostic vectorizer).

        """
        return None

    def get_output_dim(self) -> Callable[[int], int]:
        """Get the output dimension.

        Returns:
            Function mapping input dimension to output dimension.
            Output dimension = input_dim * number_of_pooling_methods.

        """
        num_pooling = 1 if isinstance(self._pooling, str) else len(self._pooling)
        return lambda input_dim: input_dim * num_pooling


@BaseSequenceVectorizer.register("cnn")
class CnnSequenceVectorizer(BaseSequenceVectorizer):
    """CNN-based sequence vectorizer using multiple n-gram filters.

    This vectorizer applies multiple 1D convolutions with different kernel sizes
    (n-gram filters) to capture local patterns of different lengths. Max pooling
    is applied over each filter's output to create a fixed-size representation.

    Based on "A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks
    for Sentence Classification" by Zhang and Wallace (2016).

    Args:
        input_dim: Input embedding dimension.
        num_filters: Number of filters for each n-gram size.
        ngram_filter_sizes: Tuple of n-gram sizes for convolution filters.
                           Default is (2, 3, 4, 5) for bigrams through 5-grams.
        conv_layer_activation: Activation function after convolution. Default is ReLU.
        output_dim: Optional output dimension. If provided, applies linear projection
                   after concatenating filter outputs.

    Example:
        >>> from formed.integrations.torch.modules.vectorizers import CnnSequenceVectorizer
        >>>
        >>> # Standard CNN with multiple n-gram filters
        >>> vectorizer = CnnSequenceVectorizer(
        ...     input_dim=128,
        ...     num_filters=100,
        ...     ngram_filter_sizes=(2, 3, 4, 5)
        ... )
        >>> # Output dim = 100 * 4 = 400
        >>>
        >>> # With output projection
        >>> vectorizer = CnnSequenceVectorizer(
        ...     input_dim=128,
        ...     num_filters=100,
        ...     ngram_filter_sizes=(3, 4, 5),
        ...     output_dim=256
        ... )
        >>>
        >>> # Custom activation
        >>> import torch.nn as nn
        >>> vectorizer = CnnSequenceVectorizer(
        ...     input_dim=128,
        ...     num_filters=50,
        ...     ngram_filter_sizes=(2, 3),
        ...     conv_layer_activation=nn.Tanh()
        ... )

    Note:
        - Properly handles padding masks to avoid max-pooling over padding positions
        - Output dimension without projection: num_filters * len(ngram_filter_sizes)
        - Each filter extracts patterns of a specific n-gram size

    """

    def __init__(
        self,
        input_dim: int,
        num_filters: int,
        ngram_filter_sizes: Sequence[int] = (2, 3, 4, 5),
        conv_layer_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation or torch.nn.ReLU()

        self._convolution_layers = [
            torch.nn.Conv1d(
                in_channels=self._input_dim,
                out_channels=self._num_filters,
                kernel_size=ngram_size,
            )
            for ngram_size in self._ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module("conv_layer_%d" % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        self.projection_layer: Optional[torch.nn.Linear]
        self._output_dim: int
        if output_dim:
            self.projection_layer = torch.nn.Linear(maxpool_output_dim, output_dim)
            self._output_dim = output_dim
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self) -> int:
        """Get the expected input dimension.

        Returns:
            Input embedding dimension.

        """
        return self._input_dim

    def get_output_dim(self) -> int:
        """Get the output dimension.

        Returns:
            Output vector dimension (num_filters * len(ngram_filter_sizes) or custom output_dim).

        """
        return self._output_dim

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Vectorize sequence using CNN with multiple n-gram filters.

        Args:
            inputs: Input embeddings of shape (batch_size, seq_len, input_dim).
            mask: Optional attention mask of shape (batch_size, seq_len).
                 True indicates valid positions, False indicates padding.

        Returns:
            Vectorized output of shape (batch_size, output_dim).

        """
        if mask is not None:
            inputs = inputs * mask.unsqueeze(-1)
        else:
            mask = torch.ones(*inputs.size()[:-1], device=inputs.device).bool()

        inputs = torch.transpose(inputs, 1, 2)

        filter_outputs = []
        batch_size = inputs.shape[0]
        last_unmasked_inputs = mask.sum(dim=1).unsqueeze(dim=-1)  # Shape: (batch_size, 1)
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, "conv_layer_{}".format(i))
            pool_length = inputs.shape[2] - convolution_layer.kernel_size[0] + 1

            activations = self._activation(convolution_layer(inputs))

            indices = (
                torch.arange(pool_length, device=activations.device).unsqueeze(0).expand(batch_size, pool_length)
            )  # Shape: (batch_size, pool_length)
            activations_mask = indices.ge(
                last_unmasked_inputs - convolution_layer.kernel_size[0] + 1
            )  # Shape: (batch_size, pool_length)
            activations_mask = activations_mask.unsqueeze(1).expand_as(
                activations
            )  # Shape: (batch_size, num_filters, pool_length)

            activations = activations + (
                activations_mask * min_value_of_dtype(activations.dtype)
            )  # Shape: (batch_size, pool_length)

            # Pick out the max filters
            filter_outputs.append(activations.max(dim=2)[0])

        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]
        maxpool_output[maxpool_output == min_value_of_dtype(maxpool_output.dtype)] = 0.0

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result


@BaseSequenceVectorizer.register("self_attentive")
class SelfAttentiveSequenceVectorizer(BaseSequenceVectorizer):
    """Self-attentive sequence vectorizer using learned attention weights.

    This vectorizer uses learned attention mechanisms to compute weighted averages
    of sequence embeddings. Multiple attention heads can be used to capture different
    aspects of the sequence.

    Based on "A Structured Self-attentive Sentence Embedding" by Lin et al. (2017).

    Args:
        input_dim: Input embedding dimension. Must be divisible by num_heads.
        num_heads: Number of attention heads. Each head learns different attention patterns.
        hidden_dims: Hidden dimensions for the attention scoring network.
                    Empty tuple means direct scoring without hidden layers.

    Example:
        >>> from formed.integrations.torch.modules.vectorizers import (
        ...     SelfAttentiveSequenceVectorizer
        ... )
        >>>
        >>> # Single attention head
        >>> vectorizer = SelfAttentiveSequenceVectorizer(
        ...     input_dim=128,
        ...     num_heads=1
        ... )
        >>>
        >>> # Multiple attention heads
        >>> vectorizer = SelfAttentiveSequenceVectorizer(
        ...     input_dim=128,
        ...     num_heads=4
        ... )
        >>>
        >>> # With hidden layers in attention scorer
        >>> vectorizer = SelfAttentiveSequenceVectorizer(
        ...     input_dim=128,
        ...     num_heads=2,
        ...     hidden_dims=(64,)
        ... )

    Note:
        - Each attention head operates on input_dim // num_heads dimensions
        - Outputs are concatenated across heads to preserve input dimension
        - Properly handles padding masks via masked softmax

    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 1,
        hidden_dims: Sequence[int] = (),
    ) -> None:
        assert input_dim % num_heads == 0, "Input dimension must be divisible by number of heads."

        super().__init__()
        self._input_dim = input_dim
        self._num_heads = num_heads
        self._head_dim = input_dim // num_heads

        self._scorers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    FeedForward(
                        input_dim=self._head_dim,
                        hidden_dims=hidden_dims,
                    ),
                    torch.nn.Linear(hidden_dims[-1], 1),
                )
                if hidden_dims
                else torch.nn.Linear(self._head_dim, 1)
                for _ in range(num_heads)
            ]
        )

    def get_input_dim(self) -> int:
        """Get the expected input dimension.

        Returns:
            Input embedding dimension.

        """
        return self._input_dim

    def get_output_dim(self) -> int:
        """Get the output dimension.

        Returns:
            Output dimension (same as input dimension).

        """
        return self._input_dim

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vectorize sequence using self-attention.

        Args:
            inputs: Input embeddings of shape (batch_size, seq_len, input_dim).
            mask: Optional attention mask of shape (batch_size, seq_len).
                 True indicates valid positions, False indicates padding.

        Returns:
            Vectorized output of shape (batch_size, input_dim).

        """
        if mask is None:
            mask = inputs.new_ones(inputs.size()[:-1], dtype=torch.bool)

        if mask.dim() == 2:
            # Shape: (batch_size, seq_len, 1)
            mask = mask.unsqueeze(-1)

        inputs = inputs * mask.float()

        head_outputs = []
        for head_index in range(self._num_heads):
            scorer = self._scorers[head_index]

            # Shape: (batch_size, seq_len, head_dim)
            head_input = inputs[..., head_index * self._head_dim : (head_index + 1) * self._head_dim]
            # Shape: (batch_size, seq_len, 1)
            attn_weights = masked_softmax(scorer(head_input), mask, dim=1)

            # Shape: (batch_size, head_dim)
            head_output = (attn_weights * head_input).sum(dim=1)
            head_outputs.append(head_output)

        # Shape: (batch_size, input_dim)
        output = torch.cat(head_outputs, dim=-1)
        return output


@BaseSequenceVectorizer.register("concat")
class ConcatSequenceVectorizer(BaseSequenceVectorizer):
    """Concatenates outputs from multiple sequence vectorizers.

    Applies multiple vectorizers to the same input sequence and concatenates
    their outputs along the feature dimension. This allows combining different
    vectorization strategies (e.g., mean pooling + max pooling + attention).

    Args:
        vectorizers: List of vectorizers to apply in parallel.
                    All vectorizers receive the same input sequence.

    Example:
        >>> from formed.integrations.torch.modules.vectorizers import (
        ...     ConcatSequenceVectorizer,
        ...     BagOfEmbeddingsSequenceVectorizer,
        ...     SelfAttentiveSequenceVectorizer
        ... )
        >>>
        >>> # Combine mean pooling and max pooling
        >>> vectorizers = [
        ...     BagOfEmbeddingsSequenceVectorizer(pooling="mean"),
        ...     BagOfEmbeddingsSequenceVectorizer(pooling="max"),
        ... ]
        >>> vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)
        >>>
        >>> # Combine pooling and attention
        >>> vectorizers = [
        ...     BagOfEmbeddingsSequenceVectorizer(pooling="mean"),
        ...     SelfAttentiveSequenceVectorizer(input_dim=128, num_heads=2),
        ... ]
        >>> vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)

    Note:
        - Output dimension is the sum of all vectorizer output dimensions
        - Handles both fixed and dynamic output dimensions from vectorizers

    """

    def __init__(self, vectorizers: Sequence[BaseSequenceVectorizer]) -> None:
        super().__init__()
        self._vectorizers = torch.nn.ModuleList(vectorizers)

    def get_input_dim(self) -> int | None:
        """Get the expected input dimension.

        Returns:
            First non-None input dimension from vectorizers, or None if all are None.

        """
        input_dims = [v.get_input_dim() for v in cast(Sequence[BaseSequenceVectorizer], self._vectorizers)]
        return next((dim for dim in input_dims if dim is not None), None)

    def get_output_dim(self) -> int | Callable[[int], int]:
        """Get the output dimension.

        Returns:
            Sum of all vectorizer output dimensions if all are fixed integers,
            otherwise a function that computes the sum given an input dimension.

        """
        input_dims = [v.get_output_dim() for v in cast(Sequence[BaseSequenceVectorizer], self._vectorizers)]
        if all(isinstance(dim, int) for dim in input_dims):
            return sum(cast(int, dim) for dim in input_dims)

        def _get_output_dim(input_dim: int) -> int:
            total_dim = 0
            for dim in input_dims:
                if isinstance(dim, int):
                    total_dim += dim
                else:
                    total_dim += dim(input_dim)
            return total_dim

        return _get_output_dim

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vectorize sequence by concatenating multiple vectorizer outputs.

        Args:
            inputs: Input embeddings of shape (batch_size, seq_len, input_dim).
            mask: Optional attention mask of shape (batch_size, seq_len).

        Returns:
            Concatenated vectors of shape (batch_size, output_dim).

        """
        vectors = [vectorizer(inputs, mask=mask) for vectorizer in self._vectorizers]
        return torch.cat(vectors, dim=-1)
