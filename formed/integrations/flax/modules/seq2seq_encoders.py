from collections.abc import Callable
from typing import Optional, Union, cast

import jax
from colt import Registrable
from flax import nnx

from .position_encoders import PositionEncoder


class Seq2SeqEncoder(nnx.Module, Registrable):
    def __call__(
        self,
        inputs: jax.Array,
        *,
        seq_lengths: Optional[jax.Array] = None,
    ) -> jax.Array:
        raise NotImplementedError


@Seq2SeqEncoder.register("rnn")
class RNNSeq2SeqEncoder(Seq2SeqEncoder):
    class _BidirectionalProjection(nnx.Module):
        def __init__(
            self,
            forward_cell: nnx.RNNCellBase,
            backward_cell: nnx.RNNCellBase,
            rngs: nnx.Rngs,
        ) -> None:
            in_features = int(getattr(forward_cell, "in_features"))
            forwrad_features = int(getattr(forward_cell, "hidden_features"))
            backward_features = int(getattr(backward_cell, "hidden_features"))
            self.projection = nnx.Linear(forwrad_features + backward_features, in_features, rngs=rngs)

        def __call__(self, a: jax.Array, b: jax.Array) -> jax.Array:
            return self.projection(jax.numpy.concatenate([a, b], axis=-1))

    class _RNNBlock(nnx.Module):
        def __init__(
            self,
            rnn: Union[nnx.RNN, nnx.Bidirectional],
            dropout: float,
            rngs: nnx.Rngs,
        ) -> None:
            self.rnn = rnn
            self.dropout = nnx.Dropout(dropout, rngs=rngs)

        def __call__(
            self,
            inputs: jax.Array,
            *,
            seq_lengths: Optional[jax.Array] = None,
            deterministic: Optional[bool] = None,
            rngs: Optional[nnx.Rngs] = None,
        ) -> jax.Array:
            return self.dropout(
                self.rnn(inputs, seq_lengths=seq_lengths, rngs=rngs),
                deterministic=deterministic,
                rngs=rngs,
            )

    def __init__(
        self,
        cell_factory: Callable[[nnx.Rngs], nnx.RNNCellBase],
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        rngs: Union[int, nnx.Rngs] = 0,
    ) -> None:
        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=0, out_axes=0)
        def create_block(rngs: nnx.Rngs) -> RNNSeq2SeqEncoder._RNNBlock:
            rnn: Union[nnx.RNN, nnx.Bidirectional]
            if bidirectional:
                forward_cell = cell_factory(rngs)
                backward_cell = cell_factory(rngs)
                rnn = nnx.Bidirectional(
                    forward_rnn=nnx.RNN(forward_cell, rngs=rngs),
                    backward_rnn=nnx.RNN(backward_cell, rngs=rngs),
                    merge_fn=self._BidirectionalProjection(forward_cell, backward_cell, rngs=rngs),
                )
            else:
                rnn = nnx.RNN(cell_factory(rngs), rngs=rngs)
            return self._RNNBlock(rnn, dropout=dropout, rngs=rngs)

        self.num_layers = num_layers
        self.blocks = create_block(rngs)

    def __call__(
        self,
        inputs: jax.Array,
        *,
        seq_lengths: Optional[jax.Array] = None,
        deterministic: Optional[bool] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jax.Array:
        @nnx.split_rngs(splits=self.num_layers)
        @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=nnx.Carry)
        def forward(
            x: jax.Array,
            block: RNNSeq2SeqEncoder._RNNBlock,
            rngs: Optional[nnx.Rngs],
        ) -> jax.Array:
            return block(x, seq_lengths=seq_lengths, deterministic=deterministic, rngs=rngs)

        return forward(inputs, self.blocks, rngs)


@Seq2SeqEncoder.register("lstm")
class LSTMSeq2SeqEncoder(RNNSeq2SeqEncoder):
    def __init__(
        self,
        features: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        rngs: Union[int, nnx.Rngs] = 0,
    ) -> None:
        super().__init__(
            cell_factory=lambda rngs: nnx.LSTMCell(features, features, rngs=rngs),
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            rngs=rngs,
        )


@Seq2SeqEncoder.register("optimized_lstm")
class OptimizedLSTMSeq2SeqEncoder(RNNSeq2SeqEncoder):
    def __init__(
        self,
        features: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        rngs: Union[int, nnx.Rngs] = 0,
    ) -> None:
        super().__init__(
            cell_factory=lambda rngs: nnx.OptimizedLSTMCell(features, features, rngs=rngs),
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            rngs=rngs,
        )


@Seq2SeqEncoder.register("gru")
class GRUSeq2SeqEncoder(RNNSeq2SeqEncoder):
    def __init__(
        self,
        features: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        rngs: Union[int, nnx.Rngs] = 0,
    ) -> None:
        super().__init__(
            cell_factory=lambda rngs: nnx.GRUCell(features, features, rngs=rngs),
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            rngs=rngs,
        )


@Seq2SeqEncoder.register("transformer")
class TransformerSeq2SeqEncoder(Seq2SeqEncoder):
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
            rngs: nnx.Rngs,
        ) -> None:
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
            seq_lengths: Optional[jax.Array] = None,
            deterministic: Optional[bool] = None,
            rngs: Optional[nnx.Rngs] = None,
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
        feedworward_features: Optional[int] = None,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
        position_encoder: Optional[PositionEncoder] = None,
        rngs: Union[int, nnx.Rngs] = 0,
    ) -> None:
        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=0, out_axes=0)
        def create_block(rngs: nnx.Rngs) -> TransformerSeq2SeqEncoder._TransformerBlock:
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
        self.blocks = create_block(rngs)
        self.position_encoder = position_encoder

    def __call__(
        self,
        inputs: jax.Array,
        *,
        seq_lengths: Optional[jax.Array] = None,
        deterministic: Optional[bool] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jax.Array:
        @nnx.split_rngs(splits=self.num_layers)
        @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=nnx.Carry)
        def forward(
            x: jax.Array,
            block: TransformerSeq2SeqEncoder._TransformerBlock,
            rngs: Optional[nnx.Rngs],
        ) -> jax.Array:
            if mask is not None:
                x = x * mask
            return block(x, seq_lengths=seq_lengths, deterministic=deterministic, rngs=rngs)

        mask: Optional[jax.Array] = None
        if seq_lengths is not None:
            mask = (jax.numpy.arange(inputs.shape[-2])[None, :] < seq_lengths[:, None])[..., None]
        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs)

        output = forward(inputs, self.blocks, rngs)
        if mask is not None:
            output = output * mask
        return output
