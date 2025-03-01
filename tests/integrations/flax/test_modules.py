from collections.abc import Callable
from functools import partial

import jax
import pytest

from formed.integrations.flax.modules import (
    FeedForward,
    GRUSeq2SeqEncoder,
    LSTMSeq2SeqEncoder,
    OptimizedLSTMSeq2SeqEncoder,
    RNNSeq2SeqEncoder,
    SinusoidalPositionEncoder,
    TransformerSeq2SeqEncoder,
)


class TestFeedForward:
    def test_forward(self) -> None:
        model = FeedForward(features=4, num_layers=2)
        x = jax.numpy.ones((8, 4))
        y = model(x)
        assert y.shape == (8, 4)


class TestSeq2SeqEncoder:
    @staticmethod
    @pytest.mark.parametrize(
        "encoder_type",
        [
            LSTMSeq2SeqEncoder,
            OptimizedLSTMSeq2SeqEncoder,
            GRUSeq2SeqEncoder,
        ],
    )
    def test_rnn_seq2seq_encoder(encoder_type: Callable[..., RNNSeq2SeqEncoder]) -> None:
        encoder = encoder_type(4)
        rng = jax.random.PRNGKey(0)
        inputs = jax.random.normal(rng, (2, 3, 4))
        output = encoder(inputs)
        assert output.shape == (2, 3, 4)

    @staticmethod
    @pytest.mark.parametrize(
        "encoder_type",
        [
            TransformerSeq2SeqEncoder,
            partial(TransformerSeq2SeqEncoder, num_layers=3),
            partial(TransformerSeq2SeqEncoder, position_encoder=SinusoidalPositionEncoder()),
        ],
    )
    def test_transformer_seq2seq_encoder(encoder_type: Callable[..., TransformerSeq2SeqEncoder]) -> None:
        encoder = encoder_type(4, 2)
        rng = jax.random.PRNGKey(0)
        inputs = jax.random.normal(rng, (2, 3, 4))
        output = encoder(inputs)
        assert output.shape == (2, 3, 4)

    @staticmethod
    def test_transformer_seq2seq_encoder_with_seq_lengths() -> None:
        encoder = TransformerSeq2SeqEncoder(4, 2)
        inputs = jax.numpy.ones((2, 3, 4))
        seq_lengths = jax.numpy.array([2, 3])
        output = jax.numpy.abs(encoder(inputs, seq_lengths=seq_lengths))
        assert output.shape == (2, 3, 4)
        assert (output.sum(2) != 0).sum(1).tolist() == [2, 3]
