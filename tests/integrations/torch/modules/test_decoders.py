"""Tests for sequence decoder interfaces."""

import pytest
import torch

from formed.integrations.torch.modules import (
    BaseSequenceDecoder,
    BaseSequenceDecoderStateInitializer,
    LSTMSequenceDecoder,
    LSTMSequenceDecoderStateInitializer,
    RotaryPositionalEncoder,
    SinusoidalPositionalEncoder,
    TransformerSequenceDecoder,
    TransformerSequenceDecoderStateInitializer,
)
from formed.integrations.torch.modules.states import ReorderableState


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


class TestTransformerSequenceDecoder:
    def _decoder(self, **kwargs: object) -> TransformerSequenceDecoder:
        params: dict[str, object] = dict(input_dim=8, num_heads=2, num_layers=2, feedforward_dim=16, dropout=0.0)
        params.update(kwargs)
        decoder = TransformerSequenceDecoder(**params)  # type: ignore[arg-type]
        return decoder.eval()

    @staticmethod
    def _memory_state(memory: torch.Tensor, memory_mask: torch.Tensor | None = None):
        return TransformerSequenceDecoderStateInitializer()(memory, memory_mask)

    def test_decodes_vector_sequence_and_returns_cache(self) -> None:
        decoder = self._decoder()
        outputs, state = decoder(torch.randn(3, 5, 8))

        assert outputs.shape == (3, 5, 8)
        assert state.self_keys.shape == (2, 3, 2, 5, 4)
        assert state.self_values.shape == (2, 3, 2, 5, 4)
        assert state.cross_keys is None
        assert decoder.get_input_dim() == 8
        assert decoder.get_output_dim() == 8
        assert isinstance(state, ReorderableState)

    @pytest.mark.parametrize("attention_backend", ["sdpa", "flex"])
    def test_kv_cache_matches_full_sequence(self, attention_backend: str) -> None:
        decoder = self._decoder(attention_backend=attention_backend)
        inputs = torch.randn(2, 6, 8)

        full_outputs, full_state = decoder(inputs)
        first_outputs, first_state = decoder(inputs[:, :4])
        second_outputs, second_state = decoder(inputs[:, 4:], initial_state=first_state)

        assert torch.allclose(torch.cat([first_outputs, second_outputs], dim=1), full_outputs, atol=1e-5)
        assert torch.allclose(second_state.self_keys, full_state.self_keys, atol=1e-5)
        assert second_state.self_keys.size(-2) == 6

    @pytest.mark.parametrize("cross_attention", [False, True])
    def test_flex_backend_matches_sdpa(self, cross_attention: bool) -> None:
        sdpa = self._decoder(cross_attention=cross_attention, attention_backend="sdpa")
        flex = self._decoder(cross_attention=cross_attention, attention_backend="flex")
        flex.load_state_dict(sdpa.state_dict())  # identical weights; only the kernel differs

        inputs = torch.randn(2, 5, 8)
        kwargs = {}
        if cross_attention:
            memory_mask = torch.ones(2, 7, dtype=torch.bool)
            memory_mask[0, 5:] = False
            kwargs["initial_state"] = self._memory_state(torch.randn(2, 7, 8), memory_mask)

        sdpa_outputs, _ = sdpa(inputs, **kwargs)
        flex_outputs, _ = flex(inputs, **kwargs)

        assert torch.allclose(sdpa_outputs, flex_outputs, atol=1e-5)

    def test_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="attention_backend"):
            TransformerSequenceDecoder(
                input_dim=8,
                num_heads=2,
                num_layers=1,
                feedforward_dim=16,
                attention_backend="xla",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "positional_encoder",
        [SinusoidalPositionalEncoder(input_dim=8), RotaryPositionalEncoder(input_dim=8)],
    )
    def test_kv_cache_matches_full_sequence_with_positional_encoder(
        self,
        positional_encoder: object,
    ) -> None:
        decoder = self._decoder(positional_encoder=positional_encoder)
        inputs = torch.randn(2, 6, 8)

        full_outputs, _ = decoder(inputs)
        first_outputs, first_state = decoder(inputs[:, :4])
        second_outputs, _ = decoder(inputs[:, 4:], initial_state=first_state)

        # Offsetting the positional encoder by the cache length must reproduce
        # the absolute positions used in the full-sequence pass.
        assert torch.allclose(torch.cat([first_outputs, second_outputs], dim=1), full_outputs, atol=1e-5)

    def test_token_by_token_matches_full_sequence(self) -> None:
        decoder = self._decoder()
        inputs = torch.randn(2, 5, 8)

        full_outputs, _ = decoder(inputs)
        state = None
        step_outputs = []
        for position in range(inputs.size(1)):
            output, state = decoder(inputs[:, position : position + 1], initial_state=state)
            step_outputs.append(output)

        assert torch.allclose(torch.cat(step_outputs, dim=1), full_outputs, atol=1e-5)

    def test_state_can_reorder_and_duplicate_batch_items(self) -> None:
        decoder = self._decoder()
        _, state = decoder(torch.randn(2, 3, 8))

        reordered = state.reorder(torch.tensor([1, 0, 1]))

        assert reordered.self_keys.shape == (2, 3, 2, 3, 4)
        assert torch.equal(reordered.self_keys[:, 0], state.self_keys[:, 1])
        assert torch.equal(reordered.self_keys[:, 2], state.self_keys[:, 1])

    def test_padded_outputs_are_zeroed(self) -> None:
        decoder = self._decoder()
        inputs = torch.randn(2, 4, 8)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

        outputs, _ = decoder(inputs, mask=mask)

        assert torch.count_nonzero(outputs[0, 2:]) == 0
        assert torch.count_nonzero(outputs[1, 3:]) == 0

    def test_cross_attention_returns_memory_cache(self) -> None:
        decoder = self._decoder(cross_attention=True, memory_dim=6)
        state = self._memory_state(torch.randn(2, 7, 6))

        outputs, state = decoder(torch.randn(2, 5, 8), initial_state=state)

        assert outputs.shape == (2, 5, 8)
        assert state.cross_keys is not None
        assert state.cross_keys.shape == (2, 2, 2, 7, 4)
        assert state.memory is None  # raw memory consumed once projected

    def test_cross_attention_reuses_cached_memory_incrementally(self) -> None:
        decoder = self._decoder(cross_attention=True)
        memory = torch.randn(2, 7, 8)
        memory_mask = torch.ones(2, 7, dtype=torch.bool)
        memory_mask[0, 5:] = False
        inputs = torch.randn(2, 4, 8)

        full_outputs, _ = decoder(inputs, initial_state=self._memory_state(memory, memory_mask))
        first_outputs, first_state = decoder(inputs[:, :2], initial_state=self._memory_state(memory, memory_mask))
        # The continuation carries only the returned state: memory must come from
        # the projections cached on it during the first step.
        second_outputs, _ = decoder(inputs[:, 2:], initial_state=first_state)

        assert torch.allclose(torch.cat([first_outputs, second_outputs], dim=1), full_outputs, atol=1e-5)

    def test_cross_attention_requires_memory(self) -> None:
        decoder = self._decoder(cross_attention=True)

        with pytest.raises(ValueError, match="encoder memory"):
            decoder(torch.randn(2, 3, 8))

    def test_rejects_indivisible_head_configuration(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            TransformerSequenceDecoder(input_dim=8, num_heads=3, num_layers=1, feedforward_dim=16)

    def test_is_registered_and_typed(self) -> None:
        decoder = self._decoder()

        assert isinstance(decoder, BaseSequenceDecoder)

    def test_incremental_generation_matches_full_sequence_with_cross_attention(self) -> None:
        # Exercises the seq2seq wiring: initializer -> state -> decoder, decoding
        # one token at a time while cross-attending to encoder memory.
        decoder = self._decoder(cross_attention=True)
        memory = torch.randn(2, 6, 8)
        target = torch.randn(2, 4, 8)

        full_outputs, _ = decoder(target, initial_state=self._memory_state(memory))

        state = self._memory_state(memory)
        step_outputs = []
        for position in range(target.size(1)):
            output, state = decoder(target[:, position : position + 1], initial_state=state)
            step_outputs.append(output)

        assert torch.allclose(torch.cat(step_outputs, dim=1), full_outputs, atol=1e-5)


class TestTransformerSequenceDecoderStateInitializer:
    def test_wraps_encoder_memory_into_state(self) -> None:
        initializer = TransformerSequenceDecoderStateInitializer()
        memory = torch.randn(2, 5, 8)
        mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])

        state = initializer(memory, mask)

        assert isinstance(initializer, BaseSequenceDecoderStateInitializer)
        assert state.memory is memory
        assert state.memory_mask is not None and state.memory_mask.dtype == torch.bool
        assert state.self_keys is None  # empty self-attention cache before decoding
        assert state.cross_keys is None
