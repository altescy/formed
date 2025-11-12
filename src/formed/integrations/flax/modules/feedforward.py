"""Feed-forward neural network modules for Flax models.

This module provides feed-forward network layers with support for
multiple layers, dropout, layer normalization, and residual connections.

Key Components:
    - Block: Single feed-forward block with optional normalization and dropout
    - FeedForward: Stacked feed-forward blocks with configurable connections

Features:
    - Configurable activation functions
    - Layer normalization with custom epsilon
    - Dropout for regularization
    - Dense residual connections
    - Efficient implementation using scan

Example:
    >>> from formed.integrations.flax.modules import FeedForward
    >>> from flax import nnx
    >>> import jax.nn
    >>>
    >>> # Simple 3-layer feed-forward network
    >>> ffn = FeedForward(
    ...     features=128,
    ...     num_layers=3,
    ...     dropout=0.1,
    ...     activation=jax.nn.gelu,
    ...     rngs=nnx.Rngs(0)
    ... )
    >>>
    >>> # With dense residual connections
    >>> ffn = FeedForward(
    ...     features=128,
    ...     num_layers=3,
    ...     residual_connection="dense",
    ...     rngs=rngs
    ... )

"""

from collections.abc import Callable
from typing import Literal, Optional

import jax
from flax import nnx

from ..random import require_rngs


class Block(nnx.Module):
    """Single feed-forward block with optional normalization and dropout.

    A block consists of a linear transformation, activation, optional dropout,
    and optional layer normalization. It can also accept a residual input.

    Args:
        rngs: Random number generators.
        input_dim: Input dimension.
        output_dim: Output dimension.
        dropout: Dropout rate (0 means no dropout).
        layer_norm_eps: Layer normalization epsilon (None means no layer norm).
        activation: Activation function (default: ReLU).

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        layer_norm_eps: Optional[float] = None,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        rngs: Optional[nnx.Rngs] = None,
    ) -> None:
        rngs = rngs or require_rngs()
        self.linear = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.activation = activation
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0.0 else None
        self.layer_norm = (
            nnx.LayerNorm(output_dim, epsilon=layer_norm_eps, rngs=rngs) if layer_norm_eps is not None else None
        )

    def __call__(
        self,
        x: jax.Array,
        r: Optional[jax.Array] = None,
        *,
        deterministic: Optional[bool] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jax.Array:
        x = self.activation(self.linear(x))
        if r is not None:
            x = x + r
        if self.dropout is not None:
            x = self.dropout(x, deterministic=deterministic, rngs=rngs)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class FeedForward(nnx.Module):
    """Multi-layer feed-forward neural network.

    This module stacks multiple feed-forward blocks with configurable
    activation, dropout, normalization, and residual connections.

    Args:
        features: Hidden dimension for all layers.
        num_layers: Number of layers.
        dropout: Dropout rate applied after each activation.
        layer_norm_eps: Epsilon for layer normalization (None disables layer norm).
        activation: Activation function (default: ReLU).
        residual_connection: Type of residual connection:
            - "none": No residual connections (default)
            - "dense": Dense connections (each layer receives sum of all previous)
        rngs: Random number generators (can be int seed or nnx.Rngs).

    Example:
        >>> # Simple 3-layer network
        >>> ffn = FeedForward(features=256, num_layers=3, rngs=0)
        >>> output = ffn(x)
        >>>
        >>> # With dropout and layer norm
        >>> ffn = FeedForward(
        ...     features=256,
        ...     num_layers=3,
        ...     dropout=0.1,
        ...     layer_norm_eps=1e-6,
        ...     rngs=0
        ... )
        >>>
        >>> # With dense residual connections
        >>> ffn = FeedForward(
        ...     features=256,
        ...     num_layers=4,
        ...     residual_connection="dense",
        ...     rngs=0
        ... )

    Note:
        When using residual_connection="dense", each layer receives the
        sum of outputs from all previous layers, similar to DenseNet.

    """

    def __init__(
        self,
        features: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        layer_norm_eps: Optional[float] = None,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        residual_connection: Literal["none", "dense"] = "none",
        rngs: Optional[nnx.Rngs] = None,
    ) -> None:
        rngs = rngs or require_rngs()

        @nnx.vmap(in_axes=0, out_axes=0)
        def create_block(rngs: nnx.Rngs) -> Block:
            return Block(features, features, dropout, layer_norm_eps, activation, rngs=rngs)

        self.features = features
        self.num_layers = num_layers
        self.blocks = create_block(rngs.fork(split=num_layers))
        self.residual_connection = residual_connection

    def get_input_dim(self) -> int:
        return self.features

    def get_output_dim(self) -> int:
        return self.features

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: Optional[bool] = None,
    ) -> jax.Array:
        prev = jax.numpy.zeros_like(x) if self.residual_connection == "dense" else None

        def forward(
            inputs: tuple[jax.Array, Optional[jax.Array]], block: Block
        ) -> tuple[tuple[jax.Array, Optional[jax.Array]], None]:
            x, r = inputs
            output = block(x, r, deterministic=deterministic)
            if self.residual_connection == "dense" and r is not None:
                r = r + x
            return (output, r), None

        (output, _), _ = jax.lax.scan(
            forward,
            (x, prev),
            self.blocks,
        )

        return output
