"""Tests for encoders module."""

import pytest
import torch

from formed.integrations.torch.modules.encoders import (
    BasePositionalEncoder,
    BaseSequenceEncoder,
    FeedForwardSequenceEncoder,
    GRUSequenceEncoder,
    LearnablePositionalEncoder,
    LSTMSequenceEncoder,
    ResidualSequenceEncoder,
    RotaryPositionalEncoder,
    SinusoidalPositionalEncoder,
    StackedSequenceEncoder,
    TransformerEncoder,
)
from formed.integrations.torch.modules.feedforward import FeedForward
from formed.integrations.torch.modules.masks import (
    CausalMask,
    CombinedMask,
    SlidingWindowAttentionMask,
)


def create_encoder_instances():
    """Create various encoder instances for testing common interface."""
    return [
        pytest.param(LSTMSequenceEncoder(input_dim=16, hidden_dim=16), id="lstm"),
        pytest.param(GRUSequenceEncoder(input_dim=16, hidden_dim=16), id="gru"),
        pytest.param(LSTMSequenceEncoder(input_dim=16, hidden_dim=8, bidirectional=True), id="lstm_bidir"),
        pytest.param(GRUSequenceEncoder(input_dim=16, hidden_dim=8, bidirectional=True), id="gru_bidir"),
        pytest.param(FeedForwardSequenceEncoder(FeedForward(input_dim=16, hidden_dims=[32, 16])), id="feedforward"),
        pytest.param(ResidualSequenceEncoder(LSTMSequenceEncoder(input_dim=16, hidden_dim=16)), id="residual_lstm"),
        pytest.param(ResidualSequenceEncoder(GRUSequenceEncoder(input_dim=16, hidden_dim=16)), id="residual_gru"),
        pytest.param(
            StackedSequenceEncoder(
                [
                    LSTMSequenceEncoder(input_dim=16, hidden_dim=16),
                    GRUSequenceEncoder(input_dim=16, hidden_dim=16),
                ]
            ),
            id="stacked",
        ),
    ]


class TestBaseSequenceEncoderInterface:
    """Test common interface for all SequenceEncoder implementations."""

    @pytest.mark.parametrize("encoder", create_encoder_instances())
    def test_has_required_methods(self, encoder):
        """Test that all encoders implement required methods."""
        assert isinstance(encoder, BaseSequenceEncoder)
        assert hasattr(encoder, "forward")
        assert hasattr(encoder, "get_input_dim")
        assert hasattr(encoder, "get_output_dim")
        assert callable(encoder.forward)
        assert callable(encoder.get_input_dim)
        assert callable(encoder.get_output_dim)

    @pytest.mark.parametrize("encoder", create_encoder_instances())
    def test_dimension_methods_return_int(self, encoder):
        """Test that dimension methods return integers."""
        input_dim = encoder.get_input_dim()
        output_dim = encoder.get_output_dim()

        assert isinstance(input_dim, int)
        assert isinstance(output_dim, int)
        assert input_dim > 0
        assert output_dim > 0

    @pytest.mark.parametrize("encoder", create_encoder_instances())
    def test_forward_basic(self, encoder):
        """Test basic forward pass for all encoders."""
        batch_size, seq_len = 2, 5
        input_dim = encoder.get_input_dim()
        output_dim = encoder.get_output_dim()

        inputs = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(inputs)

        # Check output shape (may vary due to packing)
        assert output.shape[0] == batch_size
        assert output.shape[1] <= seq_len
        assert output.shape[-1] == output_dim

    @pytest.mark.parametrize("encoder", create_encoder_instances())
    def test_forward_accepts_mask(self, encoder):
        """Test that all encoders accept mask parameter."""
        batch_size, seq_len = 2, 5
        input_dim = encoder.get_input_dim()

        inputs = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.ones(batch_size, seq_len)
        mask[:, -2:] = 0

        # Skip residual encoders with masks due to pack_padded_sequence limitation
        if isinstance(encoder, ResidualSequenceEncoder):
            pytest.skip("Residual encoders with masks have known limitations with pack_padded_sequence")

        # Should not raise error
        output = encoder(inputs, mask=mask)
        assert output is not None

    @pytest.mark.parametrize("encoder", create_encoder_instances())
    def test_gradients_flow(self, encoder):
        """Test that gradients flow correctly for all encoders."""
        batch_size, seq_len = 2, 5
        input_dim = encoder.get_input_dim()

        inputs = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        output = encoder(inputs)
        loss = output.sum()
        loss.backward()

        assert inputs.grad is not None
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

    @pytest.mark.parametrize("encoder", create_encoder_instances())
    def test_deterministic_without_dropout(self, encoder):
        """Test that forward pass is deterministic when dropout is disabled."""
        batch_size, seq_len = 2, 5
        input_dim = encoder.get_input_dim()

        encoder.eval()  # Disable dropout
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output1 = encoder(inputs)
        output2 = encoder(inputs)

        assert torch.allclose(output1, output2)

    @pytest.mark.parametrize("encoder", create_encoder_instances())
    @pytest.mark.parametrize("batch_size,seq_len", [(1, 3), (2, 5), (4, 8)])
    def test_various_batch_and_sequence_sizes(self, encoder, batch_size, seq_len):
        """Test encoders with various batch and sequence sizes."""
        input_dim = encoder.get_input_dim()
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = encoder(inputs)

        assert output.shape[0] == batch_size
        assert output.shape[1] <= seq_len
        assert output.shape[-1] == encoder.get_output_dim()


class TestRNNEncoders:
    """Tests specific to LSTM and GRU encoders."""

    @pytest.mark.parametrize("encoder_cls", [LSTMSequenceEncoder, GRUSequenceEncoder])
    @pytest.mark.parametrize("input_dim,hidden_dim", [(8, 16), (16, 32)])
    def test_bidirectional_output_dim(self, encoder_cls, input_dim, hidden_dim):
        """Test output dimension with bidirectional encoders."""
        encoder = encoder_cls(input_dim=input_dim, hidden_dim=hidden_dim, bidirectional=True)

        assert encoder.get_output_dim() == hidden_dim * 2

    @pytest.mark.parametrize("encoder_cls", [LSTMSequenceEncoder, GRUSequenceEncoder])
    @pytest.mark.parametrize("bidirectional", [True, False])
    def test_bidirectional_forward(self, encoder_cls, bidirectional):
        """Test forward pass with bidirectional flag."""
        batch_size, seq_len, input_dim, hidden_dim = 2, 8, 8, 16

        encoder = encoder_cls(input_dim=input_dim, hidden_dim=hidden_dim, bidirectional=bidirectional)
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = encoder(inputs)

        expected_output_dim = hidden_dim * (2 if bidirectional else 1)
        assert output.shape == (batch_size, seq_len, expected_output_dim)

    @pytest.mark.parametrize(
        "encoder_cls,rnn_attr",
        [
            (LSTMSequenceEncoder, "lstm"),
            (GRUSequenceEncoder, "gru"),
        ],
    )
    @pytest.mark.parametrize(
        "num_layers,dropout,expected_dropout",
        [
            (1, 0.5, 0.0),  # Single layer - no dropout
            (2, 0.5, 0.5),  # Multiple layers - dropout applied
            (3, 0.3, 0.3),  # Multiple layers - dropout applied
        ],
    )
    def test_dropout_between_layers(self, encoder_cls, rnn_attr, num_layers, dropout, expected_dropout):
        """Test dropout is only applied when num_layers > 1."""
        encoder = encoder_cls(input_dim=8, hidden_dim=16, num_layers=num_layers, dropout=dropout)

        rnn_module = getattr(encoder, rnn_attr)
        assert rnn_module.dropout == expected_dropout

    @pytest.mark.parametrize("encoder_cls", [LSTMSequenceEncoder, GRUSequenceEncoder])
    def test_variable_length_sequences(self, encoder_cls):
        """Test with variable length sequences using mask and pack_padded_sequence."""
        batch_size, max_seq_len, input_dim, hidden_dim = 4, 10, 8, 16

        encoder = encoder_cls(input_dim=input_dim, hidden_dim=hidden_dim)
        inputs = torch.randn(batch_size, max_seq_len, input_dim)

        # Create mask with variable lengths
        mask = torch.ones(batch_size, max_seq_len)
        mask[0, 7:] = 0  # Length 7
        mask[1, 5:] = 0  # Length 5
        mask[2, 9:] = 0  # Length 9
        mask[3, 3:] = 0  # Length 3

        output = encoder(inputs, mask=mask)

        # With pack_padded_sequence, output length may be shorter
        assert output.shape[0] == batch_size
        assert output.shape[1] <= max_seq_len
        assert output.shape[2] == hidden_dim


class TestResidualSequenceEncoder:
    """Tests specific to ResidualSequenceEncoder."""

    def test_mismatched_dimensions_raise_error(self):
        """Test that mismatched input/output dimensions raise error."""
        base_encoder = LSTMSequenceEncoder(input_dim=8, hidden_dim=16)

        with pytest.raises(AssertionError):
            ResidualSequenceEncoder(encoder=base_encoder)

    @pytest.mark.parametrize("encoder_cls", [LSTMSequenceEncoder, GRUSequenceEncoder])
    @pytest.mark.parametrize("dim", [8, 16, 32])
    def test_forward_adds_residual(self, encoder_cls, dim):
        """Test that forward adds residual connection correctly."""
        batch_size, seq_len = 2, 8

        base_encoder = encoder_cls(input_dim=dim, hidden_dim=dim)
        encoder = ResidualSequenceEncoder(encoder=base_encoder)

        inputs = torch.randn(batch_size, seq_len, dim)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, dim)
        # Output should not be identical to input (encoder adds something)
        assert not torch.allclose(output, inputs)


class TestFeedForwardSequenceEncoder:
    """Tests specific to FeedForwardSequenceEncoder."""

    def test_applies_to_each_position(self):
        """Test that feedforward is applied to each position independently."""
        batch_size, seq_len, input_dim = 2, 5, 8

        feedforward = FeedForward(input_dim=input_dim, hidden_dims=[16, 8])
        encoder = FeedForwardSequenceEncoder(feedforward=feedforward)

        inputs = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(inputs)

        # Apply feedforward directly to first position
        direct_output = feedforward(inputs[:, 0, :])

        # Should match encoder output at first position
        assert torch.allclose(output[:, 0, :], direct_output, atol=1e-6)

    def test_mask_is_ignored(self):
        """Test that mask parameter is ignored in FeedForwardSequenceEncoder."""
        batch_size, seq_len, input_dim = 2, 8, 8

        feedforward = FeedForward(input_dim=input_dim, hidden_dims=[16, 8])
        encoder = FeedForwardSequenceEncoder(feedforward=feedforward)

        inputs = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.ones(batch_size, seq_len)
        mask[:, -3:] = 0

        # Mask should not affect output
        output_with_mask = encoder(inputs, mask=mask)
        output_without_mask = encoder(inputs, mask=None)

        assert torch.equal(output_with_mask, output_without_mask)


class TestStackedSequenceEncoder:
    """Tests specific to StackedSequenceEncoder."""

    @pytest.mark.parametrize("num_encoders", [2, 3, 4])
    def test_multiple_encoders(self, num_encoders):
        """Test with multiple stacked encoders."""
        batch_size, seq_len, dim = 2, 8, 16

        encoders = []
        for _ in range(num_encoders):
            encoders.append(LSTMSequenceEncoder(input_dim=dim, hidden_dim=dim))

        stacked = StackedSequenceEncoder(encoders=encoders)

        assert stacked.get_input_dim() == dim
        assert stacked.get_output_dim() == dim

        inputs = torch.randn(batch_size, seq_len, dim)
        output = stacked(inputs)

        assert output.shape == (batch_size, seq_len, dim)

    def test_with_residual_encoder(self):
        """Test stacking with residual connections."""
        batch_size, seq_len, dim = 2, 8, 16

        encoder1 = LSTMSequenceEncoder(input_dim=dim, hidden_dim=dim)
        residual1 = ResidualSequenceEncoder(encoder=encoder1)

        encoder2 = LSTMSequenceEncoder(input_dim=dim, hidden_dim=dim)
        residual2 = ResidualSequenceEncoder(encoder=encoder2)

        stacked = StackedSequenceEncoder(encoders=[residual1, residual2])

        inputs = torch.randn(batch_size, seq_len, dim)
        output = stacked(inputs)

        assert output.shape == (batch_size, seq_len, dim)

    def test_forward_with_mask(self):
        """Test forward pass with mask and pack_padded_sequence behavior."""
        batch_size, seq_len = 2, 8

        encoder1 = LSTMSequenceEncoder(input_dim=8, hidden_dim=16)
        encoder2 = GRUSequenceEncoder(input_dim=16, hidden_dim=32)
        stacked = StackedSequenceEncoder(encoders=[encoder1, encoder2])

        inputs = torch.randn(batch_size, seq_len, 8)
        mask = torch.ones(batch_size, seq_len)
        mask[:, -3:] = 0

        output = stacked(inputs, mask=mask)

        # With pack_padded_sequence, output length may be shorter
        assert output.shape[0] == batch_size
        assert output.shape[1] <= seq_len
        assert output.shape[2] == 32


def create_positional_encoder_instances():
    """Create various positional encoder instances for testing common interface."""
    return [
        pytest.param(SinusoidalPositionalEncoder(input_dim=64), id="sinusoidal"),
        pytest.param(SinusoidalPositionalEncoder(input_dim=64, dropout=0.1), id="sinusoidal_dropout"),
        pytest.param(RotaryPositionalEncoder(input_dim=64), id="rotary"),
        pytest.param(RotaryPositionalEncoder(input_dim=128, max_len=1024), id="rotary_long"),
        pytest.param(LearnablePositionalEncoder(input_dim=64), id="learnable"),
        pytest.param(LearnablePositionalEncoder(input_dim=64, dropout=0.1), id="learnable_dropout"),
    ]


class TestBasePositionalEncoderInterface:
    """Test common interface for all PositionalEncoder implementations."""

    @pytest.mark.parametrize("encoder", create_positional_encoder_instances())
    def test_has_required_methods(self, encoder):
        """Test that all encoders implement required methods."""
        assert isinstance(encoder, BasePositionalEncoder)
        assert hasattr(encoder, "forward")
        assert hasattr(encoder, "get_input_dim")
        assert hasattr(encoder, "get_output_dim")
        assert callable(encoder.forward)
        assert callable(encoder.get_input_dim)
        assert callable(encoder.get_output_dim)

    @pytest.mark.parametrize("encoder", create_positional_encoder_instances())
    def test_dimension_methods_return_int(self, encoder):
        """Test that dimension methods return integers."""
        input_dim = encoder.get_input_dim()
        output_dim = encoder.get_output_dim()

        assert isinstance(input_dim, int)
        assert isinstance(output_dim, int)
        assert input_dim > 0
        assert output_dim > 0

    @pytest.mark.parametrize("encoder", create_positional_encoder_instances())
    def test_input_output_dims_match(self, encoder):
        """Test that input and output dimensions match for positional encoders."""
        assert encoder.get_input_dim() == encoder.get_output_dim()

    @pytest.mark.parametrize("encoder", create_positional_encoder_instances())
    def test_forward_basic(self, encoder):
        """Test basic forward pass for all encoders."""
        batch_size, seq_len = 2, 10
        input_dim = encoder.get_input_dim()

        inputs = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, input_dim)

    @pytest.mark.parametrize("encoder", create_positional_encoder_instances())
    def test_forward_accepts_mask(self, encoder):
        """Test that all encoders accept mask parameter."""
        batch_size, seq_len = 2, 10
        input_dim = encoder.get_input_dim()

        inputs = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.ones(batch_size, seq_len)
        mask[:, -3:] = 0

        # Should not raise error
        output = encoder(inputs, mask=mask)
        assert output is not None
        assert output.shape == (batch_size, seq_len, input_dim)

    @pytest.mark.parametrize("encoder", create_positional_encoder_instances())
    def test_gradients_flow(self, encoder):
        """Test that gradients flow correctly for all encoders."""
        batch_size, seq_len = 2, 10
        input_dim = encoder.get_input_dim()

        inputs = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        output = encoder(inputs)
        loss = output.sum()
        loss.backward()

        assert inputs.grad is not None
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

    @pytest.mark.parametrize("encoder", create_positional_encoder_instances())
    def test_deterministic_without_dropout(self, encoder):
        """Test that forward pass is deterministic when dropout is disabled."""
        batch_size, seq_len = 2, 10
        input_dim = encoder.get_input_dim()

        encoder.eval()  # Disable dropout
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output1 = encoder(inputs)
        output2 = encoder(inputs)

        assert torch.allclose(output1, output2)

    @pytest.mark.parametrize("encoder", create_positional_encoder_instances())
    @pytest.mark.parametrize("batch_size,seq_len", [(1, 5), (2, 10), (4, 20)])
    def test_various_batch_and_sequence_sizes(self, encoder, batch_size, seq_len):
        """Test encoders with various batch and sequence sizes."""
        input_dim = encoder.get_input_dim()
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, input_dim)


class TestSinusoidalPositionalEncoder:
    """Tests specific to SinusoidalPositionalEncoder."""

    @pytest.mark.parametrize("input_dim", [64, 128, 256])
    def test_output_shape(self, input_dim):
        """Test output shape matches input shape."""
        batch_size, seq_len = 2, 10
        encoder = SinusoidalPositionalEncoder(input_dim=input_dim)

        inputs = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, input_dim)

    def test_positional_encoding_is_deterministic(self):
        """Test that positional encoding is the same across batches."""
        batch_size, seq_len, input_dim = 3, 10, 64
        encoder = SinusoidalPositionalEncoder(input_dim=input_dim, dropout=0.0)
        encoder.eval()

        inputs = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(inputs)

        # The difference between batches should only come from input differences
        # (positional encoding is the same across all batches)
        # Use appropriate tolerance for floating-point arithmetic
        for i in range(1, batch_size):
            input_diff = inputs[i] - inputs[0]
            output_diff = output[i] - output[0]
            assert torch.allclose(input_diff, output_diff, rtol=1e-5, atol=1e-6)

    def test_max_len_constraint(self):
        """Test that sequence length exceeding max_len raises error."""
        max_len = 50
        encoder = SinusoidalPositionalEncoder(input_dim=64, max_len=max_len)

        # Should work for seq_len <= max_len
        inputs_ok = torch.randn(2, max_len, 64)
        output = encoder(inputs_ok)
        assert output.shape[1] == max_len

        # Should raise for seq_len > max_len
        inputs_too_long = torch.randn(2, max_len + 1, 64)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            encoder(inputs_too_long)

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
    def test_dropout(self, dropout):
        """Test dropout is applied correctly."""
        batch_size, seq_len, input_dim = 2, 10, 64
        encoder = SinusoidalPositionalEncoder(input_dim=input_dim, dropout=dropout)

        if dropout > 0:
            encoder.train()
            inputs = torch.randn(batch_size, seq_len, input_dim)

            # Multiple forward passes should give different results with dropout
            output1 = encoder(inputs)
            output2 = encoder(inputs)
            assert not torch.allclose(output1, output2)
        else:
            encoder.eval()
            inputs = torch.randn(batch_size, seq_len, input_dim)
            output1 = encoder(inputs)
            output2 = encoder(inputs)
            assert torch.allclose(output1, output2)


class TestRotaryPositionalEncoder:
    """Tests specific to RotaryPositionalEncoder."""

    @pytest.mark.parametrize("input_dim", [64, 128, 256])
    def test_output_shape(self, input_dim):
        """Test output shape matches input shape."""
        batch_size, seq_len = 2, 10
        encoder = RotaryPositionalEncoder(input_dim=input_dim)

        inputs = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, input_dim)

    def test_requires_even_dimension(self):
        """Test that odd input_dim raises error."""
        with pytest.raises(ValueError, match="input_dim must be even"):
            RotaryPositionalEncoder(input_dim=63)

    def test_rotary_encoding_is_deterministic(self):
        """Test that rotary encoding is deterministic."""
        batch_size, seq_len, input_dim = 2, 10, 64
        encoder = RotaryPositionalEncoder(input_dim=input_dim)
        encoder.eval()

        inputs = torch.randn(batch_size, seq_len, input_dim)

        output1 = encoder(inputs)
        output2 = encoder(inputs)

        assert torch.allclose(output1, output2)

    def test_max_len_constraint(self):
        """Test that sequence length exceeding max_len raises error."""
        max_len = 50
        encoder = RotaryPositionalEncoder(input_dim=64, max_len=max_len)

        # Should work for seq_len <= max_len
        inputs_ok = torch.randn(2, max_len, 64)
        output = encoder(inputs_ok)
        assert output.shape[1] == max_len

        # Should raise for seq_len > max_len
        inputs_too_long = torch.randn(2, max_len + 1, 64)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            encoder(inputs_too_long)

    def test_rotation_changes_representation(self):
        """Test that RoPE actually changes the representation."""
        batch_size, seq_len, input_dim = 2, 10, 64
        encoder = RotaryPositionalEncoder(input_dim=input_dim)

        inputs = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(inputs)

        # Output should be different from input (rotation is applied)
        assert not torch.allclose(inputs, output)

    @pytest.mark.parametrize("base", [10000.0, 1000.0, 100000.0])
    def test_different_base_values(self, base):
        """Test encoder works with different base values."""
        batch_size, seq_len, input_dim = 2, 10, 64
        encoder = RotaryPositionalEncoder(input_dim=input_dim, base=base)

        inputs = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, input_dim)


class TestLearnablePositionalEncoder:
    """Tests specific to LearnablePositionalEncoder."""

    @pytest.mark.parametrize("input_dim", [64, 128, 256])
    def test_output_shape(self, input_dim):
        """Test output shape matches input shape."""
        batch_size, seq_len = 2, 10
        encoder = LearnablePositionalEncoder(input_dim=input_dim)

        inputs = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, input_dim)

    def test_has_learnable_parameters(self):
        """Test that encoder has learnable position embeddings."""
        encoder = LearnablePositionalEncoder(input_dim=64, max_len=100)

        # Should have learnable parameters
        params = list(encoder.parameters())
        assert len(params) > 0

        # Position embeddings should be learnable
        assert encoder.position_embeddings.weight.requires_grad

    def test_embeddings_are_trained(self):
        """Test that position embeddings are updated during training."""
        batch_size, seq_len, input_dim = 2, 10, 64
        encoder = LearnablePositionalEncoder(input_dim=input_dim, max_len=100)

        # Store initial embeddings
        initial_embeddings = encoder.position_embeddings.weight.clone()

        # Simulate training step
        inputs = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(inputs)
        loss = output.sum()
        loss.backward()

        # Embeddings should have gradients
        assert encoder.position_embeddings.weight.grad is not None

        # Simulate optimizer step
        with torch.no_grad():
            encoder.position_embeddings.weight -= 0.1 * encoder.position_embeddings.weight.grad

        # Embeddings should have changed
        assert not torch.allclose(initial_embeddings, encoder.position_embeddings.weight)

    def test_max_len_constraint(self):
        """Test that sequence length exceeding max_len raises error."""
        max_len = 50
        encoder = LearnablePositionalEncoder(input_dim=64, max_len=max_len)

        # Should work for seq_len <= max_len
        inputs_ok = torch.randn(2, max_len, 64)
        output = encoder(inputs_ok)
        assert output.shape[1] == max_len

        # Should raise for seq_len > max_len
        inputs_too_long = torch.randn(2, max_len + 1, 64)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            encoder(inputs_too_long)

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
    def test_dropout(self, dropout):
        """Test dropout is applied correctly."""
        batch_size, seq_len, input_dim = 2, 10, 64
        encoder = LearnablePositionalEncoder(input_dim=input_dim, dropout=dropout, max_len=100)

        if dropout > 0:
            encoder.train()
            inputs = torch.randn(batch_size, seq_len, input_dim)

            # Multiple forward passes should give different results with dropout
            output1 = encoder(inputs)
            output2 = encoder(inputs)
            assert not torch.allclose(output1, output2)
        else:
            encoder.eval()
            inputs = torch.randn(batch_size, seq_len, input_dim)
            output1 = encoder(inputs)
            output2 = encoder(inputs)
            assert torch.allclose(output1, output2)

    def test_position_ids_are_sequential(self):
        """Test that position IDs are sequential from 0 to seq_len-1."""
        batch_size, seq_len, input_dim = 2, 10, 64
        encoder = LearnablePositionalEncoder(input_dim=input_dim, max_len=100)

        inputs = torch.randn(batch_size, seq_len, input_dim)

        # Manually compute what the output should be
        position_ids = torch.arange(seq_len)
        expected_pos_emb = encoder.position_embeddings(position_ids)

        # Check that position embeddings were added correctly (ignoring dropout)
        encoder.eval()
        output_no_dropout = encoder(inputs)
        for b in range(batch_size):
            expected_output = inputs[b] + expected_pos_emb
            assert torch.allclose(output_no_dropout[b], expected_output)


class TestTransformerEncoder:
    """Tests specific to TransformerEncoder."""

    def test_basic_forward(self):
        """Test basic forward pass without masks."""
        encoder = TransformerEncoder(
            input_dim=16,
            num_heads=2,
            num_layers=1,
            feedforward_dim=32,
            dropout=0.1,
        )

        batch_size, seq_len = 2, 5
        inputs = torch.randn(batch_size, seq_len, 16)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, 16)

    def test_with_positional_encoder(self):
        """Test transformer with positional encoding."""
        encoder = TransformerEncoder(
            input_dim=16,
            num_heads=2,
            num_layers=1,
            feedforward_dim=32,
            dropout=0.1,
            positional_encoder=SinusoidalPositionalEncoder(input_dim=16),
        )

        batch_size, seq_len = 2, 5
        inputs = torch.randn(batch_size, seq_len, 16)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, 16)

    def test_with_causal_mask(self):
        """Test transformer with causal masking."""
        encoder = TransformerEncoder(
            input_dim=16,
            num_heads=2,
            num_layers=1,
            feedforward_dim=32,
            positional_encoder=SinusoidalPositionalEncoder(input_dim=16),
            attention_mask=CausalMask(),
        )

        batch_size, seq_len = 2, 5
        inputs = torch.randn(batch_size, seq_len, 16)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, 16)

    def test_with_padding_mask(self):
        """Test transformer with padding mask from input."""
        encoder = TransformerEncoder(
            input_dim=16,
            num_heads=2,
            num_layers=1,
            feedforward_dim=32,
            positional_encoder=SinusoidalPositionalEncoder(input_dim=16),
        )

        batch_size, seq_len = 2, 5
        inputs = torch.randn(batch_size, seq_len, 16)
        mask = torch.ones(batch_size, seq_len)
        mask[:, -2:] = 0  # Mask last 2 positions

        output = encoder(inputs, mask=mask)

        assert output.shape == (batch_size, seq_len, 16)

    def test_with_sliding_window_mask(self):
        """Test transformer with sliding window attention mask."""
        encoder = TransformerEncoder(
            input_dim=16,
            num_heads=2,
            num_layers=1,
            feedforward_dim=32,
            positional_encoder=SinusoidalPositionalEncoder(input_dim=16),
            attention_mask=SlidingWindowAttentionMask(window_size=2),
        )

        batch_size, seq_len = 2, 5
        inputs = torch.randn(batch_size, seq_len, 16)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, 16)

    def test_with_combined_masks(self):
        """Test transformer with combined masks (e.g., multiple CausalMasks)."""
        # Note: This is a simple test with just CausalMask for demonstration
        # In practice, you would combine different structural masks
        encoder = TransformerEncoder(
            input_dim=16,
            num_heads=2,
            num_layers=1,
            feedforward_dim=32,
            positional_encoder=SinusoidalPositionalEncoder(input_dim=16),
            attention_mask=CombinedMask(masks=[CausalMask()]),
        )

        batch_size, seq_len = 2, 5
        inputs = torch.randn(batch_size, seq_len, 16)
        mask = torch.ones(batch_size, seq_len)
        mask[0, -1:] = 0  # Mask last 1 position in first batch
        mask[1, -2:] = 0  # Mask last 2 positions in second batch

        output = encoder(inputs, mask=mask)

        assert output.shape == (batch_size, seq_len, 16)

    def test_deterministic_without_dropout(self):
        """Test that forward pass is deterministic when dropout is disabled."""
        encoder = TransformerEncoder(
            input_dim=16,
            num_heads=2,
            num_layers=1,
            feedforward_dim=32,
            dropout=0.0,
            positional_encoder=SinusoidalPositionalEncoder(input_dim=16, dropout=0.0),
        )
        encoder.eval()

        batch_size, seq_len = 2, 5
        inputs = torch.randn(batch_size, seq_len, 16)

        output1 = encoder(inputs)
        output2 = encoder(inputs)

        assert torch.allclose(output1, output2)

    def test_gradients_flow(self):
        """Test that gradients flow correctly."""
        encoder = TransformerEncoder(
            input_dim=16,
            num_heads=2,
            num_layers=1,
            feedforward_dim=32,
            positional_encoder=SinusoidalPositionalEncoder(input_dim=16),
        )

        batch_size, seq_len = 2, 5
        inputs = torch.randn(batch_size, seq_len, 16, requires_grad=True)
        output = encoder(inputs)
        loss = output.sum()
        loss.backward()

        assert inputs.grad is not None
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_dimension_methods(self):
        """Test get_input_dim and get_output_dim methods."""
        encoder = TransformerEncoder(
            input_dim=32,
            num_heads=2,
            num_layers=1,
            feedforward_dim=64,
        )

        assert encoder.get_input_dim() == 32
        assert encoder.get_output_dim() == 32

    def test_batch_first_false(self):
        """Test transformer with batch_first=False."""
        encoder = TransformerEncoder(
            input_dim=16,
            num_heads=2,
            num_layers=1,
            feedforward_dim=32,
            batch_first=False,
        )

        seq_len, batch_size = 5, 2
        inputs = torch.randn(seq_len, batch_size, 16)
        output = encoder(inputs)

        assert output.shape == (seq_len, batch_size, 16)

    @pytest.mark.parametrize("num_heads", [2, 4])
    def test_different_num_heads(self, num_heads):
        """Test transformer with different numbers of attention heads."""
        input_dim = 16
        encoder = TransformerEncoder(
            input_dim=input_dim,
            num_heads=num_heads,
            num_layers=1,
            feedforward_dim=32,
        )

        batch_size, seq_len = 2, 5
        inputs = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, input_dim)

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_different_num_layers(self, num_layers):
        """Test transformer with different numbers of layers."""
        encoder = TransformerEncoder(
            input_dim=16,
            num_heads=2,
            num_layers=num_layers,
            feedforward_dim=32,
        )

        batch_size, seq_len = 2, 5
        inputs = torch.randn(batch_size, seq_len, 16)
        output = encoder(inputs)

        assert output.shape == (batch_size, seq_len, 16)

    def test_is_base_sequence_encoder(self):
        """Test that TransformerEncoder inherits from BaseSequenceEncoder."""
        encoder = TransformerEncoder(
            input_dim=16,
            num_heads=2,
            num_layers=1,
            feedforward_dim=32,
        )

        assert isinstance(encoder, BaseSequenceEncoder)
