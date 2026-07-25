"""Tests for sequence decoder interfaces."""

import pytest
import torch

from formed.integrations.torch.modules import (
    BaseSequenceDecoder,
    BaseSequenceDecoderStateInitializer,
    LSTMSequenceDecoder,
    LSTMSequenceDecoderStateInitializer,
)


class TestLSTMSequenceDecoderStateInitializer:
    def test_projects_last_valid_encoder_vector(self) -> None:
        initializer = LSTMSequenceDecoderStateInitializer(input_dim=4, hidden_dim=3, num_layers=2)
        outputs = torch.randn(2, 5, 4)
        mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, False]])

        state = initializer(outputs, mask)

        assert state.hidden.shape == (2, 2, 3)
        assert state.cell.shape == (2, 2, 3)
        assert isinstance(initializer, BaseSequenceDecoderStateInitializer)

    def test_rejects_empty_source_sequence(self) -> None:
        initializer = LSTMSequenceDecoderStateInitializer(input_dim=4, hidden_dim=3)

        with pytest.raises(ValueError, match="empty sequence"):
            initializer(torch.randn(1, 2, 4), torch.zeros(1, 2, dtype=torch.bool))


class StatelessSequenceDecoder(BaseSequenceDecoder[None, None]):
    """Minimal stateless decoder used to exercise the base interface."""

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        initial_state: None = None,
        params: None = None,
    ) -> tuple[torch.Tensor, None]:
        if mask is not None:
            inputs = inputs * mask.unsqueeze(-1)
        return inputs, None


def test_sequence_decoder_accepts_vector_sequences_and_mask() -> None:
    decoder = StatelessSequenceDecoder()
    inputs = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    outputs, state = decoder(inputs, mask=mask)

    assert outputs.shape == inputs.shape
    assert torch.equal(outputs, inputs * mask.unsqueeze(-1))
    assert state is None


class TestLSTMSequenceDecoder:
    def test_decodes_vector_sequence_and_returns_state(self) -> None:
        decoder = LSTMSequenceDecoder(input_dim=4, hidden_dim=6, num_layers=2)
        inputs = torch.randn(3, 5, 4)

        outputs, state = decoder(inputs)

        assert outputs.shape == (3, 5, 6)
        assert state.hidden.shape == (2, 3, 6)
        assert state.cell.shape == (2, 3, 6)
        assert decoder.get_input_dim() == 4
        assert decoder.get_output_dim() == 6

    def test_initial_state_supports_incremental_decoding(self) -> None:
        decoder = LSTMSequenceDecoder(input_dim=4, hidden_dim=6)
        inputs = torch.randn(2, 5, 4)

        full_outputs, full_state = decoder(inputs)
        first_outputs, first_state = decoder(inputs[:, :3])
        second_outputs, incremental_state = decoder(inputs[:, 3:], initial_state=first_state)

        assert torch.allclose(torch.cat([first_outputs, second_outputs], dim=1), full_outputs)
        assert torch.allclose(incremental_state.hidden, full_state.hidden)
        assert torch.allclose(incremental_state.cell, full_state.cell)

    def test_mask_preserves_input_sequence_length(self) -> None:
        decoder = LSTMSequenceDecoder(input_dim=4, hidden_dim=6)
        inputs = torch.randn(2, 5, 4)
        mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])

        outputs, state = decoder(inputs, mask=mask)

        assert outputs.shape == (2, 5, 6)
        assert torch.count_nonzero(outputs[0, 3:]) == 0
        assert torch.count_nonzero(outputs[1, 2:]) == 0
        assert state.hidden.shape == (1, 2, 6)
        assert state.cell.shape == (1, 2, 6)

    def test_state_can_reorder_and_duplicate_batch_items(self) -> None:
        decoder = LSTMSequenceDecoder(input_dim=4, hidden_dim=6)
        _, state = decoder(torch.randn(2, 3, 4))

        reordered = state.reorder(torch.tensor([1, 0, 1]))

        assert reordered.hidden.shape == (1, 3, 6)
        assert reordered.cell.shape == (1, 3, 6)
        assert torch.equal(reordered.hidden[:, 0], state.hidden[:, 1])
        assert torch.equal(reordered.hidden[:, 1], state.hidden[:, 0])
        assert torch.equal(reordered.hidden[:, 2], state.hidden[:, 1])

    def test_single_layer_disables_recurrent_dropout(self) -> None:
        decoder = LSTMSequenceDecoder(input_dim=4, hidden_dim=6, dropout=0.5)

        assert decoder.lstm.dropout == 0.0
