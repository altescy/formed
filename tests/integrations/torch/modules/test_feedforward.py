"""Tests for FeedForward module."""

import torch
import torch.nn as nn

from formed.integrations.torch.modules.feedforward import FeedForward


class TestFeedForward:
    def test_initialization(self):
        """Test FeedForward initialization."""
        input_dim = 128
        hidden_dims = [256, 512, 256]
        ffn = FeedForward(input_dim=input_dim, hidden_dims=hidden_dims)

        assert ffn.get_input_dim() == input_dim
        assert ffn.get_output_dim() == hidden_dims[-1]
        assert len(ffn._layers) == len(hidden_dims)

    def test_forward_pass(self):
        """Test forward pass with various input shapes."""
        input_dim = 128
        hidden_dims = [256, 128]
        batch_size = 16
        seq_len = 32

        ffn = FeedForward(input_dim=input_dim, hidden_dims=hidden_dims)

        # Test 2D input (batch_size, input_dim)
        inputs_2d = torch.randn(batch_size, input_dim)
        output_2d = ffn(inputs_2d)
        assert output_2d.shape == (batch_size, hidden_dims[-1])

        # Test 3D input (batch_size, seq_len, input_dim)
        inputs_3d = torch.randn(batch_size, seq_len, input_dim)
        output_3d = ffn(inputs_3d)
        assert output_3d.shape == (batch_size, seq_len, hidden_dims[-1])

    def test_single_layer(self):
        """Test FeedForward with a single hidden layer."""
        input_dim = 64
        hidden_dims = [32]
        batch_size = 8

        ffn = FeedForward(input_dim=input_dim, hidden_dims=hidden_dims)
        inputs = torch.randn(batch_size, input_dim)
        output = ffn(inputs)

        assert output.shape == (batch_size, hidden_dims[-1])

    def test_dropout(self):
        """Test FeedForward with dropout."""
        input_dim = 128
        hidden_dims = [256, 128]
        dropout = 0.5
        batch_size = 16

        ffn = FeedForward(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)

        # In training mode, dropout should be active
        ffn.train()
        inputs = torch.randn(batch_size, input_dim)
        output_train_1 = ffn(inputs)
        output_train_2 = ffn(inputs)

        # Outputs should be different due to dropout randomness
        assert not torch.allclose(output_train_1, output_train_2)

        # In eval mode, dropout should be disabled
        ffn.eval()
        output_eval_1 = ffn(inputs)
        output_eval_2 = ffn(inputs)

        # Outputs should be identical
        assert torch.allclose(output_eval_1, output_eval_2)

    def test_custom_activation(self):
        """Test FeedForward with custom activation function."""
        input_dim = 64
        hidden_dims = [128, 64]
        batch_size = 8

        # Test with GELU activation
        ffn_gelu = FeedForward(input_dim=input_dim, hidden_dims=hidden_dims, activation=nn.GELU())
        inputs = torch.randn(batch_size, input_dim)
        output_gelu = ffn_gelu(inputs)
        assert output_gelu.shape == (batch_size, hidden_dims[-1])

        # Test with Tanh activation
        ffn_tanh = FeedForward(input_dim=input_dim, hidden_dims=hidden_dims, activation=nn.Tanh())
        output_tanh = ffn_tanh(inputs)
        assert output_tanh.shape == (batch_size, hidden_dims[-1])

        # Outputs should be different with different activations
        assert not torch.allclose(output_gelu, output_tanh)

    def test_multiple_layers(self):
        """Test FeedForward with multiple hidden layers."""
        input_dim = 128
        hidden_dims = [256, 512, 256, 128, 64]
        batch_size = 4

        ffn = FeedForward(input_dim=input_dim, hidden_dims=hidden_dims)
        inputs = torch.randn(batch_size, input_dim)
        output = ffn(inputs)

        assert output.shape == (batch_size, hidden_dims[-1])
        assert len(ffn._layers) == len(hidden_dims)

    def test_dimension_transformation(self):
        """Test dimension transformation through the network."""
        input_dim = 100
        hidden_dims = [200, 50, 25]
        batch_size = 8

        ffn = FeedForward(input_dim=input_dim, hidden_dims=hidden_dims)
        inputs = torch.randn(batch_size, input_dim)
        output = ffn(inputs)

        # Verify output dimension
        assert output.shape == (batch_size, 25)

    def test_zero_dropout(self):
        """Test FeedForward with zero dropout (default)."""
        input_dim = 64
        hidden_dims = [128, 64]
        batch_size = 8

        ffn = FeedForward(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0)
        inputs = torch.randn(batch_size, input_dim)

        # With zero dropout, outputs should be deterministic
        output_1 = ffn(inputs)
        output_2 = ffn(inputs)

        assert torch.allclose(output_1, output_2)

    def test_backward_pass(self):
        """Test that gradients flow correctly through the network."""
        input_dim = 64
        hidden_dims = [128, 64]
        batch_size = 8

        ffn = FeedForward(input_dim=input_dim, hidden_dims=hidden_dims)
        inputs = torch.randn(batch_size, input_dim, requires_grad=True)

        output = ffn(inputs)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for inputs
        assert inputs.grad is not None
        assert inputs.grad.shape == inputs.shape

        # Check that gradients are computed for all parameters
        for param in ffn.parameters():
            assert param.grad is not None
