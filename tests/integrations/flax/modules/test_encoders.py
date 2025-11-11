import jax
from flax import nnx

from formed.integrations.flax.modules.encoders import (
    GRUSequenceEncoder,
    LearnablePositionEncoder,
    LSTMSequenceEncoder,
    OptimizedLSTMSequenceEncoder,
    SinusoidalPositionEncoder,
    TransformerSequenceEncoder,
)


class TestSinusoidalPositionEncoder:
    def test_basic_encoding(self) -> None:
        """Test basic sinusoidal position encoding."""
        encoder = SinusoidalPositionEncoder(max_length=512)

        inputs = jax.numpy.ones((2, 10, 64))
        output = encoder(inputs)

        assert output.shape == (2, 10, 64)
        # Output should be different from input due to added positional encoding
        assert not jax.numpy.allclose(output, inputs)

    def test_max_length(self) -> None:
        """Test with sequences up to max length."""
        encoder = SinusoidalPositionEncoder(max_length=100)

        inputs = jax.numpy.ones((1, 100, 32))
        output = encoder(inputs)

        assert output.shape == (1, 100, 32)

    def test_different_feature_dims(self) -> None:
        """Test with different feature dimensions."""
        encoder = SinusoidalPositionEncoder(max_length=512)

        for features in [32, 64, 128, 256]:
            inputs = jax.numpy.ones((2, 10, features))
            output = encoder(inputs)
            assert output.shape == (2, 10, features)

    def test_caching(self) -> None:
        """Test that encodings are cached."""
        encoder = SinusoidalPositionEncoder(max_length=512)

        inputs1 = jax.numpy.ones((2, 10, 64))
        output1 = encoder(inputs1)

        inputs2 = jax.numpy.ones((3, 10, 64))
        output2 = encoder(inputs2)

        # Should use same cached encodings for same feature dimension
        assert output1.shape == (2, 10, 64)
        assert output2.shape == (3, 10, 64)


class TestLearnablePositionEncoder:
    def test_basic_encoding(self) -> None:
        """Test basic learnable position encoding."""
        rngs = nnx.Rngs(0)
        encoder = LearnablePositionEncoder(features=64, max_length=512, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))
        output = encoder(inputs)

        assert output.shape == (2, 10, 64)

    def test_max_length(self) -> None:
        """Test with sequences up to max length."""
        rngs = nnx.Rngs(0)
        encoder = LearnablePositionEncoder(features=32, max_length=100, rngs=rngs)

        inputs = jax.numpy.ones((1, 100, 32))
        output = encoder(inputs)

        assert output.shape == (1, 100, 32)

    def test_different_feature_dims(self) -> None:
        """Test with different feature dimensions."""
        for features in [32, 64, 128]:
            rngs = nnx.Rngs(0)
            encoder = LearnablePositionEncoder(features=features, max_length=512, rngs=rngs)

            inputs = jax.numpy.ones((2, 10, features))
            output = encoder(inputs)
            assert output.shape == (2, 10, features)


class TestLSTMSequenceEncoder:
    def test_single_layer(self) -> None:
        """Test LSTM encoder with single layer."""
        rngs = nnx.Rngs(0)
        encoder = LSTMSequenceEncoder(features=64, num_layers=1, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))
        output = encoder(inputs)

        assert output.shape == (2, 10, 64)
        assert encoder.get_input_dim() == 64
        assert encoder.get_output_dim() == 64

    def test_multiple_layers(self) -> None:
        """Test LSTM encoder with multiple layers."""
        rngs = nnx.Rngs(0)
        encoder = LSTMSequenceEncoder(features=128, num_layers=3, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 128))
        output = encoder(inputs)

        assert output.shape == (2, 10, 128)

    def test_bidirectional(self) -> None:
        """Test bidirectional LSTM encoder."""
        rngs = nnx.Rngs(0)
        encoder = LSTMSequenceEncoder(features=64, num_layers=2, bidirectional=True, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))
        output = encoder(inputs)

        assert output.shape == (2, 10, 64)

    def test_with_dropout(self) -> None:
        """Test LSTM encoder with dropout."""
        rngs = nnx.Rngs(0)
        encoder = LSTMSequenceEncoder(features=64, num_layers=2, dropout=0.5, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))

        # Training mode
        output_train = encoder(inputs, deterministic=False, rngs=nnx.Rngs(1))
        # Eval mode
        output_eval = encoder(inputs, deterministic=True)

        assert output_train.shape == (2, 10, 64)
        assert output_eval.shape == (2, 10, 64)

    def test_with_mask(self) -> None:
        """Test LSTM encoder with mask."""
        rngs = nnx.Rngs(0)
        encoder = LSTMSequenceEncoder(features=64, num_layers=2, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))
        mask = jax.numpy.array([[True] * 7 + [False] * 3, [True] * 5 + [False] * 5])

        output = encoder(inputs, mask=mask)

        assert output.shape == (2, 10, 64)


class TestOptimizedLSTMSequenceEncoder:
    def test_basic_forward(self) -> None:
        """Test optimized LSTM encoder."""
        rngs = nnx.Rngs(0)
        encoder = OptimizedLSTMSequenceEncoder(features=64, num_layers=2, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))
        output = encoder(inputs)

        assert output.shape == (2, 10, 64)

    def test_bidirectional(self) -> None:
        """Test bidirectional optimized LSTM encoder."""
        rngs = nnx.Rngs(0)
        encoder = OptimizedLSTMSequenceEncoder(features=64, num_layers=2, bidirectional=True, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))
        output = encoder(inputs)

        assert output.shape == (2, 10, 64)


class TestGRUSequenceEncoder:
    def test_single_layer(self) -> None:
        """Test GRU encoder with single layer."""
        rngs = nnx.Rngs(0)
        encoder = GRUSequenceEncoder(features=64, num_layers=1, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))
        output = encoder(inputs)

        assert output.shape == (2, 10, 64)
        assert encoder.get_input_dim() == 64
        assert encoder.get_output_dim() == 64

    def test_multiple_layers(self) -> None:
        """Test GRU encoder with multiple layers."""
        rngs = nnx.Rngs(0)
        encoder = GRUSequenceEncoder(features=128, num_layers=3, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 128))
        output = encoder(inputs)

        assert output.shape == (2, 10, 128)

    def test_bidirectional(self) -> None:
        """Test bidirectional GRU encoder."""
        rngs = nnx.Rngs(0)
        encoder = GRUSequenceEncoder(features=64, num_layers=2, bidirectional=True, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))
        output = encoder(inputs)

        assert output.shape == (2, 10, 64)

    def test_with_dropout(self) -> None:
        """Test GRU encoder with dropout."""
        rngs = nnx.Rngs(0)
        encoder = GRUSequenceEncoder(features=64, num_layers=2, dropout=0.5, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))

        # Training mode
        output_train = encoder(inputs, deterministic=False, rngs=nnx.Rngs(1))
        # Eval mode
        output_eval = encoder(inputs, deterministic=True)

        assert output_train.shape == (2, 10, 64)
        assert output_eval.shape == (2, 10, 64)

    def test_with_mask(self) -> None:
        """Test GRU encoder with mask."""
        rngs = nnx.Rngs(0)
        encoder = GRUSequenceEncoder(features=64, num_layers=2, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))
        mask = jax.numpy.array([[True] * 7 + [False] * 3, [True] * 5 + [False] * 5])

        output = encoder(inputs, mask=mask)

        assert output.shape == (2, 10, 64)


class TestTransformerSequenceEncoder:
    def test_basic_forward(self) -> None:
        """Test basic transformer encoder."""
        rngs = nnx.Rngs(0)
        encoder = TransformerSequenceEncoder(features=128, num_heads=8, num_layers=2, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 128))
        output = encoder(inputs)

        assert output.shape == (2, 10, 128)
        assert encoder.get_input_dim() == 128
        assert encoder.get_output_dim() == 128

    def test_single_layer(self) -> None:
        """Test transformer encoder with single layer."""
        rngs = nnx.Rngs(0)
        encoder = TransformerSequenceEncoder(features=64, num_heads=4, num_layers=1, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 64))
        output = encoder(inputs)

        assert output.shape == (2, 10, 64)

    def test_multiple_layers(self) -> None:
        """Test transformer encoder with multiple layers."""
        rngs = nnx.Rngs(0)
        encoder = TransformerSequenceEncoder(features=256, num_heads=8, num_layers=6, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 256))
        output = encoder(inputs)

        assert output.shape == (2, 10, 256)

    def test_with_dropout(self) -> None:
        """Test transformer encoder with dropout."""
        rngs = nnx.Rngs(0)
        encoder = TransformerSequenceEncoder(features=128, num_heads=8, num_layers=2, dropout=0.1, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 128))

        # Training mode
        output_train = encoder(inputs, deterministic=False, rngs=nnx.Rngs(1))
        # Eval mode
        output_eval = encoder(inputs, deterministic=True)

        assert output_train.shape == (2, 10, 128)
        assert output_eval.shape == (2, 10, 128)

    def test_with_sinusoidal_position_encoder(self) -> None:
        """Test transformer with sinusoidal position encoding."""
        rngs = nnx.Rngs(0)
        position_encoder = SinusoidalPositionEncoder(max_length=512)
        encoder = TransformerSequenceEncoder(
            features=128, num_heads=8, num_layers=2, position_encoder=position_encoder, rngs=rngs
        )

        inputs = jax.numpy.ones((2, 10, 128))
        output = encoder(inputs)

        assert output.shape == (2, 10, 128)

    def test_with_learnable_position_encoder(self) -> None:
        """Test transformer with learnable position encoding."""
        rngs = nnx.Rngs(0)
        position_encoder = LearnablePositionEncoder(features=128, max_length=512, rngs=rngs)
        encoder = TransformerSequenceEncoder(
            features=128, num_heads=8, num_layers=2, position_encoder=position_encoder, rngs=rngs
        )

        inputs = jax.numpy.ones((2, 10, 128))
        output = encoder(inputs)

        assert output.shape == (2, 10, 128)

    def test_with_mask(self) -> None:
        """Test transformer encoder with mask."""
        rngs = nnx.Rngs(0)
        encoder = TransformerSequenceEncoder(features=128, num_heads=8, num_layers=2, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 128))
        mask = jax.numpy.array([[True] * 7 + [False] * 3, [True] * 5 + [False] * 5])

        output = encoder(inputs, mask=mask)

        assert output.shape == (2, 10, 128)

    def test_custom_feedforward_features(self) -> None:
        """Test transformer with custom feedforward dimension."""
        rngs = nnx.Rngs(0)
        encoder = TransformerSequenceEncoder(
            features=128, num_heads=8, num_layers=2, feedworward_features=256, rngs=rngs
        )

        inputs = jax.numpy.ones((2, 10, 128))
        output = encoder(inputs)

        assert output.shape == (2, 10, 128)

    def test_custom_activation(self) -> None:
        """Test transformer with custom activation function."""
        rngs = nnx.Rngs(0)
        encoder = TransformerSequenceEncoder(features=128, num_heads=8, num_layers=2, activation=jax.nn.relu, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 128))
        output = encoder(inputs)

        assert output.shape == (2, 10, 128)

    def test_custom_epsilon(self) -> None:
        """Test transformer with custom layer norm epsilon."""
        rngs = nnx.Rngs(0)
        encoder = TransformerSequenceEncoder(features=128, num_heads=8, num_layers=2, epsilon=1e-5, rngs=rngs)

        inputs = jax.numpy.ones((2, 10, 128))
        output = encoder(inputs)

        assert output.shape == (2, 10, 128)

    def test_different_num_heads(self) -> None:
        """Test transformer with different numbers of attention heads."""
        for num_heads in [1, 2, 4, 8]:
            rngs = nnx.Rngs(0)
            encoder = TransformerSequenceEncoder(features=128, num_heads=num_heads, num_layers=1, rngs=rngs)

            inputs = jax.numpy.ones((2, 10, 128))
            output = encoder(inputs)

            assert output.shape == (2, 10, 128)
