"""Sequence encoding modules for Flax models.

This module provides encoders that process sequential data, including
position encoders, RNN-based encoders, and transformer encoders.

Key Components:
    Position Encoders:
    - BasePositionEncoder: Abstract base for position encoding
    - SinusoidalPositionEncoder: Sinusoidal position embeddings
    - LearnablePositionEncoder: Learned position embeddings

    Sequence Encoders:
    - BaseSequenceEncoder: Abstract base for sequence encoders
    - RNNSequenceEncoder: Generic RNN encoder (LSTM, GRU, vanilla RNN)
    - LSTMSequenceEncoder: LSTM-specific encoder
    - OptimizedLSTMSequenceEncoder: Optimized LSTM encoder
    - GRUSequenceEncoder: GRU-specific encoder
    - TransformerSequenceEncoder: Transformer encoder with multi-head attention

Features:
    - Bidirectional RNN support
    - Stacked layers with dropout
    - Position encoding for transformers
    - Efficient implementation using scan and vmap
    - Masked sequence processing

Example:
    >>> from formed.integrations.flax.modules import (
    ...     LSTMSequenceEncoder,
    ...     TransformerSequenceEncoder,
    ...     SinusoidalPositionEncoder
    ... )
    >>> from flax import nnx
    >>>
    >>> # Bidirectional LSTM encoder
    >>> encoder = LSTMSequenceEncoder(
    ...     features=128,
    ...     num_layers=2,
    ...     bidirectional=True,
    ...     dropout=0.1,
    ...     rngs=nnx.Rngs(0)
    ... )
    >>>
    >>> # Transformer encoder with sinusoidal positions
    >>> encoder = TransformerSequenceEncoder(
    ...     features=128,
    ...     num_heads=8,
    ...     num_layers=6,
    ...     position_encoder=SinusoidalPositionEncoder(),
    ...     rngs=rngs
    ... )

"""

import abc
from collections.abc import Callable
from functools import lru_cache
from typing import cast

import jax
import numpy
from colt import Registrable
from flax import nnx

from ..random import require_rngs
from .feedforward import FeedForward


class BasePositionEncoder(Registrable, abc.ABC):
    """Abstract base class for position encoders.

    Position encoders add positional information to input embeddings,
    allowing models to understand token positions in sequences.

    """

    @abc.abstractmethod
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Add position encoding to inputs.

        Args:
            inputs: Input embeddings of shape (..., seq_len, features).

        Returns:
            Position-encoded embeddings of same shape as inputs.

        """
        raise NotImplementedError


@BasePositionEncoder.register("sinusoidal")
class SinusoidalPositionEncoder(BasePositionEncoder):
    """Sinusoidal position encoding from "Attention Is All You Need".

    This encoder uses sine and cosine functions of different frequencies
    to generate position embeddings without learnable parameters.

    Args:
        max_length: Maximum sequence length to support.

    Example:
        >>> encoder = SinusoidalPositionEncoder(max_length=512)
        >>> encoded = encoder(embeddings)

    Note:
        Encodings are cached for efficiency.

    """

    def __init__(self, max_length: int = 512) -> None:
        self.max_length = max_length

    @lru_cache(maxsize=1)
    def _encodings(self, features: int) -> numpy.ndarray:
        p, i = numpy.meshgrid(numpy.arange(float(self.max_length)), numpy.arange(features / 2) * 2)
        theta = (p / 1e4 ** (i / features)).T
        encodings = numpy.stack([numpy.sin(theta), numpy.cos(theta)], axis=-1).reshape((self.max_length, features))
        return encodings[None, ...]

    def __call__(self, inputs: jax.Array) -> jax.Array:
        seq_length, features = inputs.shape[-2:]
        encodings = self._encodings(features)
        return inputs + encodings[:, :seq_length, :]


@BasePositionEncoder.register("learnable")
class LearnablePositionEncoder(BasePositionEncoder):
    """Learnable position embeddings.

    This encoder uses a learned embedding matrix for position encoding,
    allowing the model to learn task-specific positional patterns.

    Args:
        features: Embedding dimension.
        rngs: Random number generators.
        max_length: Maximum sequence length to support.

    Example:
        >>> encoder = LearnablePositionEncoder(
        ...     features=128,
        ...     max_length=512,
        ...     rngs=rngs
        ... )

    """

    def __init__(
        self,
        features: int,
        *,
        max_length: int = 512,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = rngs or require_rngs()
        self.embed = nnx.Embed(max_length, features, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        seq_length, _ = inputs.shape[-2:]
        encodings = self.embed(jax.numpy.arange(seq_length))
        return inputs + encodings


class BaseSequenceEncoder(nnx.Module, Registrable, abc.ABC):
    """Abstract base class for sequence encoders.

    Sequence encoders process sequential data and output contextual
    representations for each position in the sequence.

    """

    @abc.abstractmethod
    def __call__(
        self,
        inputs: jax.Array,
        *,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Encode a sequence.

        Args:
            inputs: Input embeddings of shape (batch_size, seq_len, input_dim).
            mask: Optional attention mask of shape (batch_size, seq_len).

        Returns:
            Encoded representations of shape (batch_size, seq_len, output_dim).

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_input_dim(self) -> int | None:
        """Get the expected input dimension.

        Returns:
            Input feature dimension or None if dimension-agnostic.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_output_dim(self) -> int | Callable[[int], int]:
        """Get the output dimension.

        Returns:
            Output feature dimension or a function mapping input dim to output dim.

        """
        raise NotImplementedError


@BaseSequenceEncoder.register("rnn")
class RNNSequenceEncoder(BaseSequenceEncoder):
    """Generic RNN-based sequence encoder.

    This encoder supports various RNN cell types (LSTM, GRU, vanilla RNN)
    with optional bidirectionality, multiple layers, and dropout.

    Args:
        cell_factory: Function that creates an RNN cell given rngs.
        num_layers: Number of stacked RNN layers.
        bidirectional: Whether to use bidirectional RNN.
        dropout: Dropout rate between layers.
        rngs: Random number generators (int or nnx.Rngs).

    Note:
        This is a base class. Use LSTMSequenceEncoder, GRUSequenceEncoder,
        or OptimizedLSTMSequenceEncoder for specific cell types.

    """

    class _RNNBlock(nnx.Module):
        def __init__(
            self,
            rnn: nnx.RNN | nnx.Bidirectional,
            feedforward_layers: int,
            dropout: float,
            rngs: nnx.Rngs,
        ) -> None:
            self.rnn = rnn
            self.dropout = nnx.Dropout(dropout, rngs=rngs)

            in_features: int
            out_features: int
            if isinstance(rnn, nnx.Bidirectional):
                forward_cell = getattr(rnn.forward_rnn, "cell")
                backward_cell = getattr(rnn.backward_rnn, "cell")
                forward_features = int(getattr(forward_cell, "hidden_features"))
                backward_features = int(getattr(backward_cell, "hidden_features"))
                in_features = forward_features
                out_features = forward_features + backward_features
            else:
                cell = rnn.cell
                in_features = int(getattr(cell, "in_features"))
                out_features = int(getattr(cell, "hidden_features"))

            feedforward: FeedForward | None = None
            if feedforward_layers is not None and feedforward_layers > 0:
                feedforward = FeedForward(
                    features=out_features,
                    num_layers=feedforward_layers,
                    rngs=rngs,
                )

            self.feedforward = feedforward
            self.projection = nnx.Linear(out_features, in_features, rngs=rngs)

        def __call__(
            self,
            inputs: jax.Array,
            *,
            seq_lengths: jax.Array | None = None,
            deterministic: bool | None = None,
            rngs: nnx.Rngs | None = None,
        ) -> jax.Array:
            h = cast(jax.Array, self.rnn(inputs, seq_lengths=seq_lengths, rngs=rngs))
            if self.feedforward is not None:
                h = self.feedforward(h) + h
            h = self.projection(h)
            return self.dropout(h, deterministic=deterministic, rngs=rngs)

    def __init__(
        self,
        cell_factory: Callable[[nnx.Rngs], nnx.RNNCellBase],
        features: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        feedforward_layers: int | None = None,
        dropout: float = 0.0,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = rngs or require_rngs()

        @nnx.vmap(in_axes=0, out_axes=0)
        def create_block(rngs: nnx.Rngs) -> RNNSequenceEncoder._RNNBlock:
            rnn: nnx.RNN | nnx.Bidirectional
            if bidirectional:
                forward_cell = cell_factory(rngs)
                backward_cell = cell_factory(rngs)
                rnn = nnx.Bidirectional(
                    forward_rnn=nnx.RNN(forward_cell, rngs=rngs),
                    backward_rnn=nnx.RNN(backward_cell, rngs=rngs),
                    rngs=rngs,
                )
            else:
                rnn = nnx.RNN(cell_factory(rngs), rngs=rngs)
            return self._RNNBlock(
                rnn,
                feedforward_layers=feedforward_layers or 0,
                dropout=dropout,
                rngs=rngs,
            )

        self.blocks = create_block(rngs.fork(split=num_layers))
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.features = features

    def __call__(
        self,
        inputs: jax.Array,
        *,
        mask: jax.Array | None = None,
        deterministic: bool | None = None,
    ) -> jax.Array:
        seq_lengths: jax.Array | None = None
        if mask is not None:
            seq_lengths = jax.numpy.sum(mask.astype(jax.numpy.int32), axis=-1)

        def forward(
            x: jax.Array,
            block: RNNSequenceEncoder._RNNBlock,
        ) -> tuple[jax.Array, None]:
            return block(x, seq_lengths=seq_lengths, deterministic=deterministic), None

        output, _ = jax.lax.scan(forward, inputs, self.blocks)
        return output

    def get_input_dim(self) -> int:
        return self.features

    def get_output_dim(self) -> int:
        return self.features


@BaseSequenceEncoder.register("lstm")
class LSTMSequenceEncoder(RNNSequenceEncoder):
    """LSTM-based sequence encoder.

    Args:
        features: Hidden dimension.
        num_layers: Number of LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
        dropout: Dropout rate between layers.
        rngs: Random number generators.

    Example:
        >>> # Bidirectional 2-layer LSTM
        >>> encoder = LSTMSequenceEncoder(
        ...     features=128,
        ...     num_layers=2,
        ...     bidirectional=True,
        ...     dropout=0.1,
        ...     rngs=0
        ... )

    """

    def __init__(
        self,
        features: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        feedforward_layers: int | None = None,
        dropout: float = 0.0,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = rngs or require_rngs()
        super().__init__(
            cell_factory=lambda rngs: nnx.LSTMCell(features, features, rngs=rngs),
            features=features,
            num_layers=num_layers,
            bidirectional=bidirectional,
            feedforward_layers=feedforward_layers,
            dropout=dropout,
            rngs=rngs,
        )


@BaseSequenceEncoder.register("optimized_lstm")
class OptimizedLSTMSequenceEncoder(RNNSequenceEncoder):
    """Optimized LSTM sequence encoder.

    Uses Flax's optimized LSTM implementation for better performance.

    Args:
        features: Hidden dimension.
        num_layers: Number of LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
        dropout: Dropout rate between layers.
        rngs: Random number generators.

    """

    def __init__(
        self,
        features: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        feedforward_layers: int | None = None,
        dropout: float = 0.0,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        super().__init__(
            cell_factory=lambda rngs: nnx.OptimizedLSTMCell(features, features, rngs=rngs),
            features=features,
            num_layers=num_layers,
            bidirectional=bidirectional,
            feedforward_layers=feedforward_layers,
            dropout=dropout,
            rngs=rngs,
        )


@BaseSequenceEncoder.register("gru")
class GRUSequenceEncoder(RNNSequenceEncoder):
    """GRU-based sequence encoder.

    Args:
        features: Hidden dimension.
        num_layers: Number of GRU layers.
        bidirectional: Whether to use bidirectional GRU.
        dropout: Dropout rate between layers.
        rngs: Random number generators.

    Example:
        >>> encoder = GRUSequenceEncoder(
        ...     features=256,
        ...     num_layers=3,
        ...     bidirectional=True,
        ...     rngs=0
        ... )

    """

    def __init__(
        self,
        features: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        feedforward_layers: int | None = None,
        dropout: float = 0.0,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        super().__init__(
            cell_factory=lambda rngs: nnx.GRUCell(features, features, rngs=rngs),
            features=features,
            num_layers=num_layers,
            bidirectional=bidirectional,
            feedforward_layers=feedforward_layers,
            dropout=dropout,
            rngs=rngs,
        )


@BaseSequenceEncoder.register("transformer")
class TransformerSequenceEncoder(BaseSequenceEncoder):
    """Transformer-based sequence encoder.

    This encoder uses multi-head self-attention and feed-forward layers
    to process sequences, following the Transformer architecture.

    Args:
        features: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout: Dropout rate.
        epsilon: Layer normalization epsilon.
        feedworward_features: Feed-forward hidden dimension (defaults to 4*features).
        activation: Activation function for feed-forward layers.
        position_encoder: Optional position encoder.
        rngs: Random number generators.

    Example:
        >>> # Transformer with sinusoidal positions
        >>> encoder = TransformerSequenceEncoder(
        ...     features=512,
        ...     num_heads=8,
        ...     num_layers=6,
        ...     dropout=0.1,
        ...     position_encoder=SinusoidalPositionEncoder(),
        ...     rngs=rngs
        ... )
        >>>
        >>> # Transformer with learnable positions
        >>> encoder = TransformerSequenceEncoder(
        ...     features=512,
        ...     num_heads=8,
        ...     num_layers=6,
        ...     position_encoder=LearnablePositionEncoder(512, rngs=rngs),
        ...     rngs=rngs
        ... )

    """

    class _TransformerBlock(nnx.Module):
        def __init__(
            self,
            features: int,
            num_heads: int,
            *,
            epsilon: float,
            dropout: float,
            feedworward_features: int,
            activation: Callable[[jax.Array], jax.Array],
            rngs: nnx.Rngs | None = None,
        ) -> None:
            rngs = rngs or require_rngs()
            self.mha = nnx.MultiHeadAttention(num_heads, features, features, decode=False, rngs=rngs)
            self.feedworward = nnx.Sequential(
                nnx.Linear(features, feedworward_features, rngs=rngs),
                jax.nn.swish,
                nnx.Linear(feedworward_features, features, rngs=rngs),
            )
            self.norm1 = nnx.LayerNorm(features, epsilon=epsilon, rngs=rngs)
            self.norm2 = nnx.LayerNorm(features, epsilon=epsilon, rngs=rngs)
            self.dropout = nnx.Dropout(dropout, rngs=rngs)

        def __call__(
            self,
            inputs: jax.Array,
            *,
            mask: jax.Array | None = None,
            deterministic: bool | None = None,
            rngs: nnx.Rngs | None = None,
        ) -> jax.Array:
            output = self.norm1(self.mha(inputs, deterministic=deterministic, rngs=rngs) + inputs)
            output = self.norm2(self.feedworward(inputs) + output)
            output = self.dropout(output)
            return cast(jax.Array, output)

    def __init__(
        self,
        features: int,
        num_heads: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
        epsilon: float = 1e-6,
        feedworward_features: int | None = None,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
        position_encoder: BasePositionEncoder | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = rngs or require_rngs()

        @nnx.vmap(in_axes=0, out_axes=0)
        def create_block(rngs: nnx.Rngs) -> TransformerSequenceEncoder._TransformerBlock:
            return self._TransformerBlock(
                features=features,
                num_heads=num_heads,
                dropout=dropout,
                epsilon=epsilon,
                feedworward_features=feedworward_features or 4 * features,
                activation=activation,
                rngs=rngs,
            )

        self.num_layers = num_layers
        self.blocks = create_block(rngs.fork(split=num_layers))
        self.position_encoder = position_encoder
        self.features = features

    def __call__(
        self,
        inputs: jax.Array,
        *,
        mask: jax.Array | None = None,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        if mask is not None and mask.ndim == 2:
            mask = mask[..., None]

        def forward(
            inputs: tuple[jax.Array, jax.Array | None, nnx.Rngs | None],
            block: TransformerSequenceEncoder._TransformerBlock,
        ) -> tuple[tuple[jax.Array, jax.Array | None, nnx.Rngs | None], None]:
            x, mask, rngs = inputs
            if mask is not None:
                x = x * mask
            return (block(x, mask=mask, deterministic=deterministic), mask, rngs), None

        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs)

        (output, mask, _), _ = jax.lax.scan(forward, (inputs, mask, rngs), self.blocks)
        if mask is not None:
            output = output * mask
        return output

    def get_input_dim(self) -> int:
        return self.features

    def get_output_dim(self) -> int:
        return self.features
