from collections.abc import Callable
from typing import Optional, Union

import jax
from flax import nnx


class Block(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        layer_norm_eps: Optional[float] = None,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
    ) -> None:
        self.linear = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.activation = activation
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0.0 else None
        self.layer_norm = (
            nnx.LayerNorm(output_dim, epsilon=layer_norm_eps, rngs=rngs) if layer_norm_eps is not None else None
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.activation(self.linear(x))
        if self.dropout is not None:
            x = self.dropout(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class FeedForward(nnx.Module):
    def __init__(
        self,
        features: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        layer_norm_eps: Optional[float] = None,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        rngs: Union[int, nnx.Rngs] = 0,
    ) -> None:
        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_block(rngs: nnx.Rngs) -> Block:
            return Block(rngs, features, features, dropout, layer_norm_eps, activation)

        self.features = features
        self.num_layers = num_layers
        self.blocks = create_block(rngs)

    @property
    def input_dim(self) -> int:
        return self.features

    @property
    def output_dim(self) -> int:
        return self.features

    def __call__(self, x: jax.Array) -> jax.Array:
        @nnx.split_rngs(splits=self.num_layers)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x: jax.Array, block: Block) -> jax.Array:
            return block(x)

        return forward(x, self.blocks)
