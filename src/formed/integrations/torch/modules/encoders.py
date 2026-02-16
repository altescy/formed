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
import math
from collections.abc import Sequence
from typing import Literal, NamedTuple, Optional

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


@BaseSequenceEncoder.register("residual")
class ResidualSequenceEncoder(BaseSequenceEncoder):
    """Residual wrapper for sequence encoders.

    Adds the input to the encoder output (residual connection).
    Requires input and output dimensions to match.

    Args:
        encoder: Base encoder to wrap. Must have matching input and output dimensions.

    Example:
        >>> from formed.integrations.torch.modules.encoders import (
        ...     ResidualSequenceEncoder,
        ...     LSTMSequenceEncoder
        ... )
        >>>
        >>> # Wrap LSTM with residual connection
        >>> base_encoder = LSTMSequenceEncoder(input_dim=128, hidden_dim=128)
        >>> encoder = ResidualSequenceEncoder(encoder=base_encoder)

    """

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


@BaseSequenceEncoder.register("feedforward")
class FeedForwardSequenceEncoder(BaseSequenceEncoder):
    """Position-wise feedforward sequence encoder.

    Applies a feedforward network independently to each position in the sequence.
    The same transformation is applied at each position (no cross-position interaction).

    Args:
        feedforward: Feedforward network to apply at each position.

    Example:
        >>> from formed.integrations.torch.modules.encoders import (
        ...     FeedForwardSequenceEncoder
        ... )
        >>> from formed.integrations.torch.modules.feedforward import FeedForward
        >>>
        >>> # Apply feedforward to each position independently
        >>> feedforward = FeedForward(input_dim=128, hidden_dims=[256, 128])
        >>> encoder = FeedForwardSequenceEncoder(feedforward=feedforward)

    """

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


@BaseSequenceEncoder.register("gated_cnn")
class GatedCnnSequenceEncoder(BaseSequenceEncoder):
    """Gated Convolutional Neural Network sequence encoder.

    Uses stacked residual blocks with gated linear units (GLU) for efficient
    sequence modeling. Processes sequences in both forward and backward directions,
    then concatenates the results for bidirectional context capture.

    Based on "Language Modeling with Gated Convolutional Networks" (Dauphin et al., 2017).

    Args:
        input_dim: Input dimension.
        layers: List of layer configurations for each residual block.
                Each block is a list of Layer(kernel_size, output_dim, dilation).
        output_dim: Optional output dimension. If provided, applies linear projection.
                   Default is input_dim * 2 (concatenation of forward + backward).
        dropout: Dropout rate applied to the first convolution of each block.

    Example:
        >>> # Simple gated CNN encoder
        >>> encoder = GatedCnnSequenceEncoder(
        ...     input_dim=128,
        ...     layers=[
        ...         [GatedCnnSequenceEncoder.Layer(kernel_size=3, output_dim=128)],
        ...         [GatedCnnSequenceEncoder.Layer(kernel_size=3, output_dim=128)],
        ...     ]
        ... )
        >>>
        >>> # With dilated convolutions for larger receptive field
        >>> encoder = GatedCnnSequenceEncoder(
        ...     input_dim=128,
        ...     layers=[
        ...         [GatedCnnSequenceEncoder.Layer(kernel_size=2, output_dim=128, dilation=1)],
        ...         [GatedCnnSequenceEncoder.Layer(kernel_size=2, output_dim=128, dilation=2)],
        ...         [GatedCnnSequenceEncoder.Layer(kernel_size=2, output_dim=128, dilation=4)],
        ...     ],
        ...     output_dim=256,
        ...     dropout=0.1
        ... )

    """

    class Layer(NamedTuple):
        """Configuration for a single convolutional layer.

        Attributes:
            kernel_size: Size of the convolution kernel.
            output_dim: Output dimension of the layer. Must match input_dim
                       for residual connections to work.
            dilation: Dilation rate for the convolution. When dilation > 1,
                     kernel_size must be 2.

        """

        kernel_size: int
        output_dim: int
        dilation: int = 1

    class ResidualBlock(torch.nn.Module):
        """Residual block with gated convolutions for sequence encoding.

        Stacks multiple gated convolutional layers with residual connections.
        Supports causal masking via directional processing (forward/backward).

        Args:
            input_dim: Input dimension. Must match output dimension of all layers
                      for residual connection.
            layers: Sequence of Layer configurations defining the convolutional stack.
            direction: Direction of causal masking ("forward" or "backward").
            do_weight_norm: Whether to apply weight normalization to convolutions.
            dropout: Dropout rate applied to the first convolution.

        """

        def __init__(
            self,
            input_dim: int,
            layers: Sequence["GatedCnnSequenceEncoder.Layer"],
            direction: Literal["forward", "backward"],
            do_weight_norm: bool = True,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()

            self.dropout = dropout
            self._convolutions = torch.nn.ModuleList()
            last_dim = input_dim
            for k, layer in enumerate(layers):
                if layer.dilation == 1:
                    conv = torch.nn.Conv1d(
                        in_channels=last_dim,
                        out_channels=layer.output_dim * 2,
                        kernel_size=layer.kernel_size,
                        stride=1,
                        padding=layer[0] - 1,
                        bias=True,
                    )
                else:
                    assert layer.kernel_size == 2, "only support kernel = 2 for now"
                    conv = torch.nn.Conv1d(
                        in_channels=last_dim,
                        out_channels=layer.output_dim * 2,
                        kernel_size=layer.kernel_size,
                        stride=1,
                        padding=layer.dilation,
                        dilation=layer.dilation,
                        bias=True,
                    )

                if k == 0:
                    conv_dropout = dropout
                else:
                    conv_dropout = 0.0
                std = math.sqrt((4 * (1.0 - conv_dropout)) / (layer.kernel_size * last_dim))

                conv.weight.data.normal_(0, std=std)
                if conv.bias is not None:
                    conv.bias.data.zero_()

                if do_weight_norm:
                    conv = torch.nn.utils.weight_norm(conv, name="weight", dim=0)

                self._convolutions.append(conv)
                last_dim = layer.output_dim

            assert last_dim == input_dim

            if direction not in ("forward", "backward"):
                raise ValueError(f"invalid direction: {direction}")
            self._direction = direction

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Apply gated convolutions with residual connection.

            Args:
                inputs: Input of shape (batch_size, input_dim, seq_len).

            Returns:
                Encoded sequence with residual connection of shape (batch_size, output_dim, seq_len).

            """
            output = inputs
            sequence_length = inputs.size(2)
            for k, convolution in enumerate(self._convolutions):
                if k == 0 and self.dropout > 0:
                    output = torch.nn.functional.dropout(output, self.dropout, self.training)

                conv_out = convolution(output)

                dims_to_remove = conv_out.size(2) - sequence_length
                if dims_to_remove > 0:
                    if self._direction == "forward":
                        conv_out = conv_out.narrow(2, 0, sequence_length)
                    else:
                        conv_out = conv_out.narrow(2, dims_to_remove, sequence_length)

                output = torch.nn.functional.glu(conv_out, dim=1)

            return (output + inputs) * math.sqrt(0.5)

    def __init__(
        self,
        input_dim: int,
        layers: Sequence[Sequence["GatedCnnSequenceEncoder.Layer"]],
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self._forward_residual_blocks = torch.nn.ModuleList()
        self._backward_residual_blocks = torch.nn.ModuleList()
        self._input_dim = input_dim
        self._output_dim = output_dim or input_dim * 2

        for layer in layers:
            self._forward_residual_blocks.append(
                GatedCnnSequenceEncoder.ResidualBlock(input_dim, layer, "forward", dropout=dropout)
            )
            self._backward_residual_blocks.append(
                GatedCnnSequenceEncoder.ResidualBlock(input_dim, layer, "backward", dropout=dropout)
            )

        self._projection: Optional[torch.nn.Linear] = None
        if output_dim:
            self._projection = torch.nn.Linear(input_dim * 2, output_dim)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode input sequence using gated CNN.

        Args:
            inputs: Input of shape (batch_size, seq_len, input_dim).
            mask: Optional mask of shape (batch_size, seq_len).
                 True indicates valid positions, False indicates padding.

        Returns:
            Encoded sequence of shape (batch_size, seq_len, output_dim).

        """
        if mask is None:
            mask = torch.ones(*inputs.size()[:-1], dtype=torch.bool, device=inputs.device)
        else:
            # Ensure mask is boolean
            mask = mask.bool()

        transposed_embeddings = torch.transpose(inputs, 1, 2)
        mask_for_fill = ~mask.unsqueeze(1)

        outputs: list[torch.Tensor] = []
        for blocks in (self._forward_residual_blocks, self._backward_residual_blocks):
            out = transposed_embeddings
            for block in blocks:
                out = block(out.masked_fill(mask_for_fill, 0.0))
            outputs.append(out)

        output = torch.cat(outputs, dim=1).transpose(1, 2)
        if self._projection:
            output = self._projection(output)
        return output


@BaseSequenceEncoder.register("stacked")
class StackedSequenceEncoder(BaseSequenceEncoder):
    """Stacks multiple sequence encoders sequentially.

    Applies encoders in order, passing the output of each as input to the next.
    The output dimension of each encoder must match the input dimension of the next.

    Args:
        encoders: List of encoders to apply in sequence.
                 Each encoder's output dimension must match the next encoder's input dimension.

    Example:
        >>> from formed.integrations.torch.modules.encoders import (
        ...     StackedSequenceEncoder,
        ...     LSTMSequenceEncoder,
        ...     GRUSequenceEncoder,
        ...     ResidualSequenceEncoder
        ... )
        >>>
        >>> # Stack LSTM and GRU
        >>> encoders = [
        ...     LSTMSequenceEncoder(input_dim=128, hidden_dim=128),
        ...     GRUSequenceEncoder(input_dim=128, hidden_dim=64),
        ... ]
        >>> encoder = StackedSequenceEncoder(encoders=encoders)
        >>>
        >>> # More complex: LSTM -> Residual LSTM -> GRU
        >>> base_lstm = LSTMSequenceEncoder(input_dim=128, hidden_dim=128)
        >>> residual_lstm = ResidualSequenceEncoder(encoder=base_lstm)
        >>> gru = GRUSequenceEncoder(input_dim=128, hidden_dim=128)
        >>> encoder = StackedSequenceEncoder(encoders=[base_lstm, residual_lstm, gru])

    """

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


@BaseSequenceEncoder.register("concat")
class ConcatSequenceEncoder(BaseSequenceEncoder):
    """Concatenates outputs from multiple sequence encoders.

    Applies multiple encoders in parallel to the same input and concatenates their outputs
    along the feature dimension. All encoders receive the same input tensor.

    Args:
        encoders: List of encoders to apply in parallel.
                 All encoders must have the same input dimension.

    Example:
        >>> from formed.integrations.torch.modules.encoders import (
        ...     ConcatSequenceEncoder,
        ...     LSTMSequenceEncoder,
        ...     GRUSequenceEncoder
        ... )
        >>>
        >>> # Concatenate LSTM and GRU outputs
        >>> encoders = [
        ...     LSTMSequenceEncoder(input_dim=128, hidden_dim=64),
        ...     GRUSequenceEncoder(input_dim=128, hidden_dim=64),
        ... ]
        >>> encoder = ConcatSequenceEncoder(encoders=encoders)

    """

    def __init__(self, encoders: list[BaseSequenceEncoder]) -> None:
        super().__init__()
        self._encoders = torch.nn.ModuleList(encoders)
        self._input_dim = sum(encoder.get_input_dim() for encoder in encoders)
        self._output_dim = sum(encoder.get_output_dim() for encoder in encoders)

    def get_input_dim(self) -> int:
        """Get the expected input dimension.

        Returns:
            Sum of input dimensions across all encoders.

        """
        return self._input_dim

    def get_output_dim(self) -> int:
        """Get the output dimension.

        Returns:
            Sum of output dimensions across all encoders.

        """
        return self._output_dim

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode input sequence by concatenating outputs from all encoders.

        Args:
            inputs: Input of shape (batch_size, seq_len, input_dim).
            mask: Optional mask of shape (batch_size, seq_len).

        Returns:
            Concatenated encoded sequence of shape (batch_size, seq_len, output_dim).

        """
        outputs = []
        for encoder in self._encoders:
            outputs.append(encoder(inputs, mask=mask))
        return torch.cat(outputs, dim=-1)


@BaseSequenceEncoder.register("window_concat")
class WindowConcatSequenceEncoder(BaseSequenceEncoder):
    """Concatenates context window features for each position in the sequence.

    For each position, concatenates the embeddings from surrounding positions
    within a specified window. This creates richer positional representations
    by explicitly including local context.

    Args:
        input_dim: Input dimension.
        window_size: Size of context window on each side. If int, uses symmetric window.
                    If tuple (left, right), uses asymmetric window.
        output_dim: Optional output dimension. If provided, applies linear projection
                   to the concatenated features. Otherwise, output dimension is
                   (left_window + 1 + right_window) * input_dim.

    Example:
        >>> # Symmetric 2-position window on each side
        >>> encoder = WindowConcatSequenceEncoder(
        ...     input_dim=128,
        ...     window_size=2
        ... )
        >>>
        >>> # Asymmetric window with projection
        >>> encoder = WindowConcatSequenceEncoder(
        ...     input_dim=128,
        ...     window_size=(1, 2),
        ...     output_dim=256
        ... )

    """

    def __init__(
        self,
        input_dim: int,
        window_size: int | tuple[int, int],
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        if not all(s >= 0 for s in window_size):
            raise ValueError("Window size must be greater than or equal to zero.")
        self._input_dim = input_dim
        self._window_size = window_size
        self._projection: Optional[torch.nn.Linear] = None
        if output_dim is not None:
            self._projection = torch.nn.Linear(
                (sum(window_size) + 1) * input_dim,
                output_dim,
            )

    def get_input_dim(self) -> int:
        """Get the expected input dimension.

        Returns:
            Input dimension of the embeddings.

        """
        return self._input_dim

    def get_output_dim(self) -> int:
        """Get the output dimension.

        Returns:
            Output dimension after window concatenation and optional projection.

        """
        if self._projection is not None:
            return self._projection.out_features
        return (sum(self._window_size) + 1) * self._input_dim

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode input sequence by concatenating context windows.

        Args:
            inputs: Input of shape (batch_size, seq_len, input_dim).
            mask: Optional mask of shape (batch_size, seq_len).
                 True indicates valid positions, False indicates padding.

        Returns:
            Window-concatenated sequence of shape (batch_size, seq_len, output_dim).

        """
        batch_size, max_length, embedding_dim = inputs.size()

        if mask is None:
            mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=inputs.device)

        inputs = inputs * mask.float().unsqueeze(2)

        output = inputs
        lws, rws = self._window_size
        if lws > 0:
            pad = inputs.new_zeros((batch_size, lws, embedding_dim))
            x = torch.cat([pad, inputs], dim=1)
            x = torch.cat([x[:, offset : offset + max_length] for offset in range(lws)], dim=2)
            output = torch.cat([output, x], dim=2)
        if rws > 0:
            pad = inputs.new_zeros((batch_size, rws, embedding_dim))
            x = torch.cat([inputs, pad], dim=1)
            x = torch.cat([x[:, offset : offset + max_length] for offset in range(1, rws + 1)], dim=2)
            output = torch.cat([output, x], dim=2)

        if self._projection is not None:
            output = self._projection(output)

        return output * mask.float().unsqueeze(2)


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
