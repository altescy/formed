"""Sequence encoding modules for PyTorch models.

This module provides encoders that process sequential data, including
RNN-based encoders, positional encoders, and Transformer encoders.

Key Components:
    - BaseSequenceEncoder: Abstract base for sequence encoders
    - LSTMSequenceEncoder: LSTM-specific encoder
    - GRUSequenceEncoder: GRU-specific encoder
    - BasePositionalEncoder: Abstract base for positional encoders
    - SinusoidalPositionalEncoder: Sinusoidal positional encoding
    - RotaryPositionalEncoder: Rotary positional encoding (RoPE)
    - LearnablePositionalEncoder: Learnable positional embeddings
    - TransformerEncoder: Transformer-based encoder with configurable masking

Features:
    - Bidirectional RNN support
    - Stacked layers with dropout
    - Masked sequence processing
    - Various positional encoding strategies
    - Flexible attention masking

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
from typing import Optional

import torch
import torch.nn as nn
from colt import Registrable

from .feedforward import FeedForward
from .masks import BaseAttentionMask


class BaseSequenceEncoder(nn.Module, Registrable, abc.ABC):
    """Abstract base class for sequence encoders.

    Sequence encoders process sequential data and output encoded representations.

    """

    @abc.abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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

    def __call__(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().__call__(inputs, mask=mask)


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
        mask: Optional[torch.Tensor] = None,
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
            packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
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
        mask: Optional[torch.Tensor] = None,
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
            packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
            output, _ = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.gru(inputs)

        return output

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._hidden_dim * (2 if self._bidirectional else 1)


@BaseSequenceEncoder.register("recipe_classifier::residual")
class ResidualSequenceEncoder(BaseSequenceEncoder):
    def __init__(self, encoder: BaseSequenceEncoder) -> None:
        assert encoder.get_input_dim() == encoder.get_output_dim()

        super().__init__()
        self._encoder = encoder

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self._encoder(inputs, mask=mask)
        return inputs + encoded

    def get_input_dim(self) -> int:
        return self._encoder.get_input_dim()

    def get_output_dim(self) -> int:
        return self._encoder.get_output_dim()


@BaseSequenceEncoder.register("recipe_classifier::feedforward")
class FeedForwardSequenceEncoder(BaseSequenceEncoder):
    def __init__(self, feedforward: FeedForward) -> None:
        super().__init__()
        self._feedforward = feedforward

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask

        original_shape = inputs.shape
        flattened_inputs = inputs.view(-1, original_shape[-1])
        encoded = self._feedforward(flattened_inputs)
        return encoded.view(*original_shape[:-1], -1)

    def get_input_dim(self) -> int:
        return self._feedforward.get_input_dim()

    def get_output_dim(self) -> int:
        return self._feedforward.get_output_dim()


@BaseSequenceEncoder.register("recipe_classifier::stacked")
class StackedSequenceEncoder(BaseSequenceEncoder):
    def __init__(self, encoders: list[BaseSequenceEncoder]) -> None:
        super().__init__()
        self._encoders = torch.nn.ModuleList(encoders)
        self._input_dim = encoders[0].get_input_dim()
        self._output_dim = encoders[-1].get_output_dim()

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = inputs
        for encoder in self._encoders:
            x = encoder(x, mask=mask)
        return x

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim


class BasePositionalEncoder(nn.Module, Registrable, abc.ABC):
    """Abstract base class for positional encoders.

    Positional encoders add positional information to sequential data.

    """

    @abc.abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add positional encoding to input sequence.

        Args:
            inputs: Input sequence of shape (batch_size, seq_len, input_dim).
            mask: Optional mask of shape (batch_size, seq_len).

        Returns:
            Position-encoded sequence of shape (batch_size, seq_len, output_dim).

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

    def __call__(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().__call__(inputs, mask=mask)


@BasePositionalEncoder.register("sinusoidal")
class SinusoidalPositionalEncoder(BasePositionalEncoder):
    """Sinusoidal positional encoding.

    Uses sine and cosine functions of different frequencies to encode
    position information, as introduced in "Attention Is All You Need".

    Args:
        input_dim: Dimension of the embeddings.
        max_len: Maximum sequence length to pre-compute.
        dropout: Dropout rate to apply after adding positional encoding.

    Example:
        >>> encoder = SinusoidalPositionalEncoder(
        ...     input_dim=512,
        ...     max_len=5000,
        ...     dropout=0.1
        ... )

    """

    pe: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        max_len: int = 5000,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._max_len = max_len

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2) * (-torch.log(torch.tensor(10000.0)) / input_dim))

        pe = torch.zeros(1, max_len, input_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add sinusoidal positional encoding to inputs.

        Args:
            inputs: Input of shape (batch_size, seq_len, input_dim).
            mask: Optional mask of shape (batch_size, seq_len).

        Returns:
            Position-encoded sequence of shape (batch_size, seq_len, input_dim).

        """
        seq_len = inputs.size(1)
        if seq_len > self._max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self._max_len}")

        output = inputs + self.pe[:, :seq_len, :]
        return self.dropout(output)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim


@BasePositionalEncoder.register("rotary")
class RotaryPositionalEncoder(BasePositionalEncoder):
    """Rotary positional encoding (RoPE).

    Applies rotary position embeddings by rotating pairs of dimensions
    in the feature space, as introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    Args:
        input_dim: Dimension of the embeddings (must be even).
        max_len: Maximum sequence length to pre-compute.
        base: Base for the geometric progression (default: 10000).

    Example:
        >>> encoder = RotaryPositionalEncoder(
        ...     input_dim=512,
        ...     max_len=2048
        ... )

    """

    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        max_len: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError(f"input_dim must be even, got {input_dim}")

        self._input_dim = input_dim
        self._max_len = max_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, input_dim, 2).float() / input_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute cos and sin for max_len positions
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, :])

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply rotary positional encoding to inputs.

        Args:
            inputs: Input of shape (batch_size, seq_len, input_dim).
            mask: Optional mask of shape (batch_size, seq_len).

        Returns:
            Position-encoded sequence of shape (batch_size, seq_len, input_dim).

        """
        seq_len = inputs.size(1)
        if seq_len > self._max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self._max_len}")

        cos = self.cos_cached[:, :seq_len, :]
        sin = self.sin_cached[:, :seq_len, :]

        output = (inputs * cos) + (self._rotate_half(inputs) * sin)
        return output

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim


@BasePositionalEncoder.register("learnable")
class LearnablePositionalEncoder(BasePositionalEncoder):
    """Learnable positional embeddings.

    Uses a learnable embedding table to encode position information,
    similar to token embeddings.

    Args:
        input_dim: Dimension of the embeddings.
        max_len: Maximum sequence length (vocabulary size for positions).
        dropout: Dropout rate to apply after adding positional encoding.

    Example:
        >>> encoder = LearnablePositionalEncoder(
        ...     input_dim=512,
        ...     max_len=1024,
        ...     dropout=0.1
        ... )

    """

    def __init__(
        self,
        input_dim: int,
        max_len: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._max_len = max_len

        self.position_embeddings = nn.Embedding(max_len, input_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add learnable positional encoding to inputs.

        Args:
            inputs: Input of shape (batch_size, seq_len, input_dim).
            mask: Optional mask of shape (batch_size, seq_len).

        Returns:
            Position-encoded sequence of shape (batch_size, seq_len, input_dim).

        """
        seq_len = inputs.size(1)
        if seq_len > self._max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self._max_len}")

        position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).expand(inputs.size(0), -1)

        position_embeddings = self.position_embeddings(position_ids)
        output = inputs + position_embeddings
        return self.dropout(output)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim


@BaseSequenceEncoder.register("transformer")
class TransformerEncoder(BaseSequenceEncoder):
    """Transformer-based sequence encoder.

    Uses stacked TransformerEncoderLayers with positional encoding and
    configurable attention masking via dependency injection.

    Args:
        input_dim: Dimension of the embeddings (d_model).
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        feedforward_dim: Dimension of feedforward network.
        dropout: Dropout rate.
        positional_encoder: Optional positional encoder to add position information.
        attention_mask: Optional mask generator for self-attention.
        activation: Activation function (default: "relu").
        layer_norm_eps: Epsilon for layer normalization.
        batch_first: Whether input is batch-first (default: True).

    Example:
        >>> from formed.integrations.torch.modules.encoders import (
        ...     TransformerEncoder,
        ...     SinusoidalPositionalEncoder,
        ...     CausalMask
        ... )
        >>>
        >>> # Standard transformer encoder
        >>> encoder = TransformerEncoder(
        ...     input_dim=512,
        ...     num_heads=8,
        ...     num_layers=6,
        ...     feedforward_dim=2048,
        ...     dropout=0.1,
        ...     positional_encoder=SinusoidalPositionalEncoder(input_dim=512)
        ... )
        >>>
        >>> # Transformer with causal masking (for autoregressive tasks)
        >>> causal_encoder = TransformerEncoder(
        ...     input_dim=512,
        ...     num_heads=8,
        ...     num_layers=6,
        ...     feedforward_dim=2048,
        ...     dropout=0.1,
        ...     positional_encoder=SinusoidalPositionalEncoder(input_dim=512),
        ...     attention_mask=CausalMask()
        ... )

    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        positional_encoder: Optional[BasePositionalEncoder] = None,
        attention_mask: Optional[BaseAttentionMask] = None,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._positional_encoder = positional_encoder
        self._attention_mask = attention_mask
        self._batch_first = batch_first

        # Create transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=False,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input sequence using transformer.

        Args:
            inputs: Input of shape (batch_size, seq_len, input_dim) if batch_first=True,
                   or (seq_len, batch_size, input_dim) if batch_first=False.
            mask: Optional mask of shape (batch_size, seq_len) where 1=valid, 0=padding.

        Returns:
            Encoded sequence of same shape as input.

        """
        # Apply positional encoding if provided
        if self._positional_encoder is not None:
            inputs = self._positional_encoder(inputs, mask=mask)

        batch_size = inputs.size(0) if self._batch_first else inputs.size(1)
        seq_len = inputs.size(1) if self._batch_first else inputs.size(0)

        # Generate attention mask if generator is provided
        # All attention masks return (seq_len, seq_len) or (batch_size, seq_len, seq_len)
        src_mask = None
        if self._attention_mask is not None:
            src_mask = self._attention_mask(
                seq_len=seq_len,
                batch_size=batch_size,
                device=inputs.device,
                padding_mask=mask,
            )
            if src_mask is not None:
                src_mask = src_mask.to(inputs.device)

        # Generate key padding mask for transformer
        # This is separate from attention_mask and handles padding from input mask
        # TransformerEncoder expects True for positions to be masked
        src_key_padding_mask = None
        if mask is not None:
            # Convert mask: 1=valid -> False (not masked), 0=padding -> True (masked)
            src_key_padding_mask = ~mask.bool()

        # Apply transformer encoder
        output = self.transformer_encoder(
            inputs,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        return output

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim
