import jax
import jax.numpy as jnp
from flax import nnx

from formed.integrations.flax.modules.feedforward import Block, FeedForward


class TestBlock:
    def test_basic_forward(self) -> None:
        """Test basic forward pass through a block."""
        rngs = nnx.Rngs(0)
        block = Block(input_dim=64, output_dim=128, rngs=rngs)

        x = jnp.ones((2, 64))
        output = block(x)

        assert output.shape == (2, 128)

    def test_with_dropout(self) -> None:
        """Test block with dropout enabled."""
        rngs = nnx.Rngs(0)
        block = Block(input_dim=64, output_dim=64, dropout=0.5, rngs=rngs)

        x = jnp.ones((2, 64))

        # Training mode (dropout active)
        output_train = block(x, deterministic=False, rngs=nnx.Rngs(1))
        # Eval mode (dropout inactive)
        output_eval = block(x, deterministic=True)

        assert output_train.shape == (2, 64)
        assert output_eval.shape == (2, 64)
        # Outputs should differ due to dropout
        assert not jnp.allclose(output_train, output_eval)

    def test_with_layer_norm(self) -> None:
        """Test block with layer normalization."""
        rngs = nnx.Rngs(0)
        block = Block(input_dim=64, output_dim=64, layer_norm_eps=1e-6, rngs=rngs)

        x = jnp.ones((2, 64))
        output = block(x)

        assert output.shape == (2, 64)

    def test_with_residual(self) -> None:
        """Test block with residual connection."""
        rngs = nnx.Rngs(0)
        block = Block(input_dim=64, output_dim=64, rngs=rngs)

        x = jnp.ones((2, 64))
        residual = jnp.ones((2, 64)) * 2
        output = block(x, r=residual)

        assert output.shape == (2, 64)

    def test_custom_activation(self) -> None:
        """Test block with custom activation function."""
        rngs = nnx.Rngs(0)
        block = Block(input_dim=64, output_dim=64, activation=jax.nn.gelu, rngs=rngs)

        x = jnp.ones((2, 64))
        output = block(x)

        assert output.shape == (2, 64)


class TestFeedForward:
    def test_single_layer(self) -> None:
        """Test feed-forward network with single layer."""
        rngs = nnx.Rngs(0)
        ffn = FeedForward(features=128, num_layers=1, rngs=rngs)

        x = jnp.ones((2, 128))
        output = ffn(x)

        assert output.shape == (2, 128)
        assert ffn.get_input_dim() == 128
        assert ffn.get_output_dim() == 128

    def test_multiple_layers(self) -> None:
        """Test feed-forward network with multiple layers."""
        rngs = nnx.Rngs(0)
        ffn = FeedForward(features=128, num_layers=3, rngs=rngs)

        x = jnp.ones((2, 128))
        output = ffn(x)

        assert output.shape == (2, 128)

    def test_with_dropout(self) -> None:
        """Test feed-forward network with dropout."""
        rngs = nnx.Rngs(0)
        ffn = FeedForward(features=128, num_layers=2, dropout=0.5, rngs=rngs)

        x = jnp.ones((2, 128))

        # Training mode
        output_train = ffn(x, deterministic=False)
        # Eval mode
        output_eval = ffn(x, deterministic=True)

        assert output_train.shape == (2, 128)
        assert output_eval.shape == (2, 128)

    def test_with_layer_norm(self) -> None:
        """Test feed-forward network with layer normalization."""
        rngs = nnx.Rngs(0)
        ffn = FeedForward(features=128, num_layers=2, layer_norm_eps=1e-6, rngs=rngs)

        x = jnp.ones((2, 128))
        output = ffn(x)

        assert output.shape == (2, 128)

    def test_dense_residual_connection(self) -> None:
        """Test feed-forward network with dense residual connections."""
        rngs = nnx.Rngs(0)
        ffn = FeedForward(features=128, num_layers=3, residual_connection="dense", rngs=rngs)

        x = jnp.ones((2, 128))
        output = ffn(x)

        assert output.shape == (2, 128)

    def test_no_residual_connection(self) -> None:
        """Test feed-forward network with no residual connections."""
        rngs = nnx.Rngs(0)
        ffn = FeedForward(features=128, num_layers=3, residual_connection="none", rngs=rngs)

        x = jnp.ones((2, 128))
        output = ffn(x)

        assert output.shape == (2, 128)

    def test_custom_activation(self) -> None:
        """Test feed-forward network with custom activation function."""
        rngs = nnx.Rngs(0)
        ffn = FeedForward(features=128, num_layers=2, activation=jax.nn.gelu, rngs=rngs)

        x = jnp.ones((2, 128))
        output = ffn(x)

        assert output.shape == (2, 128)

    def test_batch_processing(self) -> None:
        """Test feed-forward network with different batch sizes."""
        rngs = nnx.Rngs(0)
        ffn = FeedForward(features=64, num_layers=2, rngs=rngs)

        # Single sample
        x1 = jnp.ones((1, 64))
        output1 = ffn(x1)
        assert output1.shape == (1, 64)

        # Small batch
        x2 = jnp.ones((4, 64))
        output2 = ffn(x2)
        assert output2.shape == (4, 64)

        # Large batch
        x3 = jnp.ones((32, 64))
        output3 = ffn(x3)
        assert output3.shape == (32, 64)

    def test_with_all_features(self) -> None:
        """Test feed-forward network with all features enabled."""
        rngs = nnx.Rngs(0)
        ffn = FeedForward(
            features=128,
            num_layers=3,
            dropout=0.1,
            layer_norm_eps=1e-6,
            activation=jax.nn.gelu,
            residual_connection="dense",
            rngs=rngs,
        )

        x = jnp.ones((2, 128))
        output = ffn(x, deterministic=False)

        assert output.shape == (2, 128)
