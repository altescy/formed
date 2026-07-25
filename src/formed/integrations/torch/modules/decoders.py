import abc
from dataclasses import dataclass
from typing import Generic, Protocol, Self, TypeAlias, TypeVar, runtime_checkable

import torch
import torch.nn as nn
from colt import Registrable

_StateT = TypeVar("_StateT")
_ParamsT = TypeVar("_ParamsT")


@runtime_checkable
class ReorderableState(Protocol):
    """Complete state required to continue decoding.

    State may include mutable runtime data, such as recurrent states or
    attention caches, and immutable conditioning data, such as encoder memory.
    """

    def reorder(self, indices: torch.Tensor) -> Self:
        """Select, reorder, or duplicate states using flattened batch indices."""
        ...


ReorderableDecoderState: TypeAlias = ReorderableState


class BaseSequenceDecoderStateInitializer(nn.Module, Registrable, Generic[_StateT], abc.ABC):
    """Build a decoder's complete initial state from encoded source vectors."""

    @abc.abstractmethod
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> _StateT:
        """Create decoder state from ``(batch, source_length, input_dim)`` vectors."""
        raise NotImplementedError


@dataclass
class LSTMDecoderState:
    """Hidden and cell states produced by an LSTM sequence decoder."""

    hidden: torch.Tensor
    cell: torch.Tensor

    def reorder(self, indices: torch.Tensor) -> Self:
        """Select, reorder, or duplicate states along the batch dimension."""
        return type(self)(
            hidden=self.hidden.index_select(1, indices),
            cell=self.cell.index_select(1, indices),
        )


@BaseSequenceDecoderStateInitializer.register("lstm")
class LSTMSequenceDecoderStateInitializer(BaseSequenceDecoderStateInitializer[LSTMDecoderState]):
    """Initialize an LSTM decoder by projecting the last valid encoder vector."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.hidden_projection = nn.Linear(input_dim, hidden_dim * num_layers)
        self.cell_projection = nn.Linear(input_dim, hidden_dim * num_layers)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> LSTMDecoderState:
        if encoder_outputs.ndim != 3:
            raise ValueError("encoder_outputs must have shape (batch_size, sequence_length, input_dim)")
        if encoder_outputs.size(-1) != self._input_dim:
            raise ValueError(f"Expected encoder output dimension {self._input_dim}, got {encoder_outputs.size(-1)}")

        if mask is None:
            last = encoder_outputs[:, -1]
        else:
            if mask.shape != encoder_outputs.shape[:2]:
                raise ValueError("mask must match the batch and sequence dimensions of encoder_outputs")
            lengths = mask.long().sum(dim=1)
            if torch.any(lengths == 0):
                raise ValueError("Cannot initialize decoder state from an empty sequence")
            indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, encoder_outputs.size(-1))
            last = encoder_outputs.gather(1, indices).squeeze(1)

        batch_size = encoder_outputs.size(0)
        hidden = self.hidden_projection(last).view(batch_size, self._num_layers, self._hidden_dim).transpose(0, 1)
        cell = self.cell_projection(last).view(batch_size, self._num_layers, self._hidden_dim).transpose(0, 1)
        return LSTMDecoderState(hidden=hidden.contiguous(), cell=cell.contiguous())


class BaseSequenceDecoder(nn.Module, Registrable, Generic[_StateT, _ParamsT], abc.ABC):
    """Abstract base class for sequence decoders.

    A SequenceDecoder transforms a sequence of input vectors into a sequence of
    output vectors. Implementations may use an optional state to support
    incremental decoding.

    Type Parameters:
        _StateT: Type of the internal state used during decoding.
        _ParamsT: Type of additional parameters used during decoding.

    """

    @abc.abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        initial_state: _StateT | None = None,
        params: _ParamsT | None = None,
    ) -> tuple[torch.Tensor, _StateT | None]:
        """Decode an input sequence.

        Args:
            inputs: Input sequence of shape `(batch_size, seq_len, input_dim)`.
            mask: Optional mask of shape `(batch_size, seq_len)`.
            initial_state: Optional initial state for the decoder.
            params: Optional additional parameters for decoding.

        Returns:
            A tuple containing:
                - Output sequence of shape `(batch_size, seq_len, output_dim)`.
                - Final state of the decoder, or `None` for stateless decoders.

        """
        raise NotImplementedError


@BaseSequenceDecoder.register("lstm")
class LSTMSequenceDecoder(BaseSequenceDecoder[LSTMDecoderState, None]):
    """LSTM-based sequence decoder.

    The decoder accepts vector sequences and returns the LSTM output sequence
    together with its hidden and cell states. Passing the returned state back as
    ``initial_state`` enables incremental decoding.

    Args:
        input_dim: Dimension of each input vector.
        hidden_dim: Dimension of each LSTM hidden state.
        num_layers: Number of recurrent layers.
        dropout: Dropout applied between recurrent layers. It is disabled when
            ``num_layers`` is one.

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        initial_state: LSTMDecoderState | None = None,
        params: None = None,
    ) -> tuple[torch.Tensor, LSTMDecoderState]:
        """Decode an input sequence and return the final recurrent state."""
        lstm_state = None if initial_state is None else (initial_state.hidden, initial_state.cell)

        if mask is None:
            outputs, (hidden, cell) = self.lstm(inputs, lstm_state)
            return outputs, LSTMDecoderState(hidden=hidden, cell=cell)

        lengths = mask.sum(dim=1).cpu()
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, (hidden, cell) = self.lstm(packed_inputs, lstm_state)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=inputs.size(1),
        )
        return outputs, LSTMDecoderState(hidden=hidden, cell=cell)

    def get_input_dim(self) -> int:
        """Return the input vector dimension."""
        return self._input_dim

    def get_output_dim(self) -> int:
        """Return the output vector dimension."""
        return self._hidden_dim
