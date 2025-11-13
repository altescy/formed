"""Sequence encoding modules for PyTorch models.

This module provides encoders that process sequential data, including
RNN-based encoders.

Key Components:
    - BaseSequenceEncoder: Abstract base for sequence encoders
    - LSTMSequenceEncoder: LSTM-specific encoder
    - GRUSequenceEncoder: GRU-specific encoder

Features:
    - Bidirectional RNN support
    - Stacked layers with dropout
    - Masked sequence processing

Example:
    >>> from formed.integrations.torch.modules import LSTMSequenceEncoder
    >>>
    >>> # Bidirectional LSTM encoder
    >>> encoder = LSTMSequenceEncoder(
    ...     input_dim=128,
    ...     hidden_dim=256,
    ...     num_layers=2,
    ...     bidirectional=True,
    ...     dropout=0.1
    ... )

"""

import abc

import torch
import torch.nn as nn
from colt import Registrable


class BaseSequenceEncoder(nn.Module, Registrable, abc.ABC):
    """Abstract base class for sequence encoders.

    Sequence encoders process sequential data and output encoded representations.

    """

    @abc.abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode input sequence.

        Args:
            inputs: Input sequence of shape (batch_size, seq_len, input_dim).
            mask: Optional mask of shape (batch_size, seq_len).

        Returns:
            Encoded sequence of shape (batch_size, seq_len, output_dim).

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_input_dim(self) -> int:
        """Get the expected input dimension."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_output_dim(self) -> int:
        """Get the output dimension."""
        raise NotImplementedError


@BaseSequenceEncoder.register("lstm")
class LSTMSequenceEncoder(BaseSequenceEncoder):
    """LSTM-based sequence encoder.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden state dimension.
        num_layers: Number of LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
        dropout: Dropout rate between layers.
        batch_first: Whether input is batch-first (default: True).

    Example:
        >>> encoder = LSTMSequenceEncoder(
        ...     input_dim=128,
        ...     hidden_dim=256,
        ...     num_layers=2,
        ...     bidirectional=True
        ... )

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=batch_first,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode input sequence.

        Args:
            inputs: Input of shape (batch_size, seq_len, input_dim).
            mask: Optional mask of shape (batch_size, seq_len).

        Returns:
            Encoded sequence of shape (batch_size, seq_len, output_dim).

        """
        if mask is not None:
            # Pack padded sequence for efficiency
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                inputs, lengths, batch_first=True, enforce_sorted=False
            )
            output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.lstm(inputs)

        return output

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._hidden_dim * (2 if self._bidirectional else 1)


@BaseSequenceEncoder.register("gru")
class GRUSequenceEncoder(BaseSequenceEncoder):
    """GRU-based sequence encoder.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden state dimension.
        num_layers: Number of GRU layers.
        bidirectional: Whether to use bidirectional GRU.
        dropout: Dropout rate between layers.
        batch_first: Whether input is batch-first (default: True).

    Example:
        >>> encoder = GRUSequenceEncoder(
        ...     input_dim=128,
        ...     hidden_dim=256,
        ...     num_layers=2,
        ...     bidirectional=True
        ... )

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=batch_first,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode input sequence.

        Args:
            inputs: Input of shape (batch_size, seq_len, input_dim).
            mask: Optional mask of shape (batch_size, seq_len).

        Returns:
            Encoded sequence of shape (batch_size, seq_len, output_dim).

        """
        if mask is not None:
            # Pack padded sequence for efficiency
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                inputs, lengths, batch_first=True, enforce_sorted=False
            )
            output, _ = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.gru(inputs)

        return output

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._hidden_dim * (2 if self._bidirectional else 1)
