import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, Self, TypeAlias, TypeVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from colt import Registrable

from .encoders import BasePositionalEncoder
from .states import ReorderableState

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

    _FLEX_ATTENTION_AVAILABLE = True
else:
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        _FLEX_ATTENTION_AVAILABLE = True
    except ImportError:  # pragma: no cover - depends on the installed torch build
        _FLEX_ATTENTION_AVAILABLE = False

AttentionBackend: TypeAlias = Literal["sdpa", "flex"]

# Mask closure passed to ``flex_attention`` via ``create_block_mask``: it returns
# ``True`` at ``(batch, head, query_index, key_index)`` positions that may attend.
_MaskMod: TypeAlias = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

_StateT = TypeVar("_StateT")
_ParamsT = TypeVar("_ParamsT")


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


@dataclass
class TransformerDecoderState:
    """State carried between incremental Transformer decoding steps.

    This is the single channel through which a Transformer decoder receives both
    its running self-attention cache and its encoder conditioning, mirroring how
    the LSTM decoder receives everything through :class:`LSTMDecoderState`. It is
    produced by :class:`TransformerSequenceDecoderStateInitializer` (from encoder
    outputs) and returned/updated by every decoder step, so beam search can
    reorder it as a whole.

    Self-attention cache:
        ``self_keys`` and ``self_values`` grow by one position per decoded token
        and are shaped ``(num_layers, batch, num_heads, length, head_dim)``. They
        are ``None`` before the first token is decoded (an empty cache).

    Cross-attention conditioning (only for encoder-decoder models):
        ``memory``/``memory_mask`` hold the raw encoder outputs (shape
        ``(batch, source_length, memory_dim)``) and their validity mask. On the
        first step the decoder projects the memory into per-layer
        ``cross_keys``/``cross_values`` (shape
        ``(num_layers, batch, num_heads, source_length, head_dim)``) and caches
        them here, reusing the projections on subsequent steps.
    """

    self_keys: torch.Tensor | None = None
    self_values: torch.Tensor | None = None
    memory: torch.Tensor | None = None
    memory_mask: torch.Tensor | None = None
    cross_keys: torch.Tensor | None = None
    cross_values: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """Number of positions currently held in the self-attention cache."""
        return 0 if self.self_keys is None else self.self_keys.size(-2)

    def reorder(self, indices: torch.Tensor) -> Self:
        """Select, reorder, or duplicate states along the batch dimension."""
        return type(self)(
            self_keys=None if self.self_keys is None else self.self_keys.index_select(1, indices),
            self_values=None if self.self_values is None else self.self_values.index_select(1, indices),
            memory=None if self.memory is None else self.memory.index_select(0, indices),
            memory_mask=None if self.memory_mask is None else self.memory_mask.index_select(0, indices),
            cross_keys=None if self.cross_keys is None else self.cross_keys.index_select(1, indices),
            cross_values=None if self.cross_values is None else self.cross_values.index_select(1, indices),
        )


class _TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder block with cache-aware self-attention.

    Attention weights and their key/value projections are managed here (rather
    than via ``nn.MultiheadAttention``) so that past keys and values can be
    cached and reused for incremental decoding, and so beam search can reorder
    them. The scaled dot-product itself is delegated to a fused kernel selected
    by ``attention_backend``: ``"sdpa"`` uses
    :func:`torch.nn.functional.scaled_dot_product_attention` with an additive
    mask, while ``"flex"`` uses :func:`~torch.nn.attention.flex_attention.flex_attention`
    with a block mask (attention dropout is not applied in the flex path).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        activation: nn.Module,
        layer_norm_eps: float,
        norm_first: bool,
        cross_attention: bool,
        memory_dim: int,
        attention_backend: AttentionBackend,
    ) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._dropout_p = dropout
        self._norm_first = norm_first
        self._cross_attention = cross_attention
        self._attention_backend = attention_backend

        self.self_projection = nn.Linear(embed_dim, 3 * embed_dim)
        self.self_output = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        if cross_attention:
            self.cross_query = nn.Linear(embed_dim, embed_dim)
            self.cross_projection = nn.Linear(memory_dim, 2 * embed_dim)
            self.cross_output = nn.Linear(embed_dim, embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _split_heads(inputs: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Reshape ``(batch, seq, dim)`` into ``(batch, num_heads, seq, head_dim)``."""
        batch_size, seq_len, dim = inputs.shape
        return inputs.view(batch_size, seq_len, num_heads, dim // num_heads).transpose(1, 2)

    @staticmethod
    def _merge_heads(inputs: torch.Tensor) -> torch.Tensor:
        """Reshape ``(batch, num_heads, seq, head_dim)`` back into ``(batch, seq, dim)``."""
        batch_size, num_heads, seq_len, head_dim = inputs.shape
        return inputs.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)

    @staticmethod
    def _causal_padding_bias(
        query_length: int,
        key_length: int,
        key_padding_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build an additive (SDPA) self-attention mask aligned to the end of the key axis.

        Query ``i`` (the ``i``-th of ``query_length`` new positions) is allowed to
        attend to key ``j`` iff ``j <= i + (key_length - query_length)``. The offset
        ``key_length - query_length`` equals the number of cached positions, so the
        same rule yields a standard causal mask when nothing is cached and the
        correct bottom-right alignment during incremental decoding. This cannot be
        delegated to ``scaled_dot_product_attention(is_causal=True)``, which aligns
        its triangular mask to the top-left and is therefore wrong once a cache is
        present. ``key_padding_mask`` marks padding keys with ``True``.
        """
        queries = torch.arange(query_length, device=device).unsqueeze(1)
        keys = torch.arange(key_length, device=device).unsqueeze(0)
        allowed = keys <= queries + (key_length - query_length)
        mask = torch.zeros(query_length, key_length, dtype=dtype, device=device)
        mask = mask.masked_fill(~allowed, float("-inf")).unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            padding = torch.zeros_like(key_padding_mask, dtype=dtype)
            padding = padding.masked_fill(key_padding_mask, float("-inf"))
            mask = mask + padding.unsqueeze(1).unsqueeze(1)
        return mask

    @staticmethod
    def _padding_bias(key_padding_mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
        """Build an additive ``(batch, 1, 1, key_length)`` (SDPA) mask for cross-attention.

        ``key_padding_mask`` marks padding keys with ``True``.
        """
        if key_padding_mask is None:
            return None
        mask = torch.zeros_like(key_padding_mask, dtype=dtype)
        return mask.masked_fill(key_padding_mask, float("-inf")).unsqueeze(1).unsqueeze(1)

    @staticmethod
    def _causal_mask_mod(offset: int, key_valid_mask: torch.Tensor | None) -> _MaskMod:
        """Build a FlexAttention ``mask_mod`` for offset-aligned causal self-attention.

        ``key_valid_mask`` (``True`` = attend, i.e. non-padding), when given, is
        captured and applied per batch element.
        """
        if key_valid_mask is None:

            def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
                return kv_idx <= q_idx + offset

        else:

            def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
                return (kv_idx <= q_idx + offset) & key_valid_mask[b, kv_idx]

        return mask_mod

    @staticmethod
    def _padding_mask_mod(key_valid_mask: torch.Tensor) -> _MaskMod:
        """Build a FlexAttention ``mask_mod`` restricting cross-attention to valid keys."""

        def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
            return key_valid_mask[b, kv_idx]

        return mask_mod

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: "torch.Tensor | BlockMask | None",
    ) -> torch.Tensor:
        """Run scaled dot-product attention through the configured backend.

        ``mask`` is a :class:`BlockMask` for the ``"flex"`` backend and an additive
        bias tensor (or ``None``) for ``"sdpa"``; the active backend determines
        which was built, so each branch narrows it accordingly.
        """
        if self._attention_backend == "flex":
            output = flex_attention(query, key, value, block_mask=cast("BlockMask | None", mask))
            return output[0] if isinstance(output, tuple) else output
        dropout_p = self._dropout_p if self.training else 0.0
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=cast("torch.Tensor | None", mask), dropout_p=dropout_p
        )

    def project_memory(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project encoder memory into this layer's cross-attention keys and values."""
        key, value = self.cross_projection(memory).chunk(2, dim=-1)
        return self._split_heads(key, self._num_heads), self._split_heads(value, self._num_heads)

    def _self_attention(
        self,
        inputs: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        mask: "torch.Tensor | BlockMask | None",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.self_projection(inputs).chunk(3, dim=-1)
        query, key, value = (self._split_heads(tensor, self._num_heads) for tensor in projected)
        if past_key is not None and past_value is not None:
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        attn = self._attend(query, key, value, mask)
        return self.self_output(self._merge_heads(attn)), key, value

    def _cross_attention_block(
        self,
        inputs: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        mask: "torch.Tensor | BlockMask | None",
    ) -> torch.Tensor:
        query = self._split_heads(self.cross_query(inputs), self._num_heads)
        attn = self._attend(query, memory_key, memory_value, mask)
        return self.cross_output(self._merge_heads(attn))

    def _feedforward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(inputs))))

    def forward(
        self,
        inputs: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        self_mask: "torch.Tensor | BlockMask | None",
        memory_key: torch.Tensor | None,
        memory_value: torch.Tensor | None,
        cross_mask: "torch.Tensor | BlockMask | None",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._norm_first:
            attn, key, value = self._self_attention(self.norm1(inputs), past_key, past_value, self_mask)
            inputs = inputs + self.dropout(attn)
            if self._cross_attention:
                assert memory_key is not None and memory_value is not None
                cross = self._cross_attention_block(self.norm2(inputs), memory_key, memory_value, cross_mask)
                inputs = inputs + self.dropout(cross)
            inputs = inputs + self.dropout(self._feedforward(self.norm3(inputs)))
        else:
            attn, key, value = self._self_attention(inputs, past_key, past_value, self_mask)
            inputs = self.norm1(inputs + self.dropout(attn))
            if self._cross_attention:
                assert memory_key is not None and memory_value is not None
                cross = self._cross_attention_block(inputs, memory_key, memory_value, cross_mask)
                inputs = self.norm2(inputs + self.dropout(cross))
            inputs = self.norm3(inputs + self.dropout(self._feedforward(inputs)))
        return inputs, key, value


@BaseSequenceDecoderStateInitializer.register("transformer")
class TransformerSequenceDecoderStateInitializer(BaseSequenceDecoderStateInitializer[TransformerDecoderState]):
    """Package encoder outputs into the initial state of a Transformer decoder.

    This is the encoder-decoder bridge for :class:`TransformerSequenceDecoder`,
    the counterpart to :class:`LSTMSequenceDecoderStateInitializer`. Unlike the
    LSTM initializer, cross-attention keys/values are projected per layer *inside*
    the decoder, so this initializer only needs to carry the raw encoder memory
    and its validity mask into :class:`TransformerDecoderState`; it holds no
    parameters. Routing memory through the state (rather than a separate decoding
    parameter) keeps the decoder swappable in the same ``initializer -> state ->
    decoder`` wiring used by the LSTM decoder, and lets beam search reorder the
    memory alongside the running cache.
    """

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> TransformerDecoderState:
        """Wrap ``(batch, source_length, memory_dim)`` encoder outputs into a decoder state."""
        return TransformerDecoderState(
            memory=encoder_outputs,
            memory_mask=None if mask is None else mask.bool(),
        )


@BaseSequenceDecoder.register("transformer")
class TransformerSequenceDecoder(BaseSequenceDecoder[TransformerDecoderState, None]):
    """Transformer sequence decoder with a reorderable key/value cache.

    The decoder applies stacked causal self-attention blocks and returns a
    :class:`TransformerDecoderState` holding the per-layer key/value cache.
    Passing that state back as ``initial_state`` continues decoding
    incrementally: only the new positions are projected, past keys and values
    are read from the cache, and the positional encoder is offset by the cache
    length. Because the cache implements
    :class:`~formed.integrations.torch.modules.states.ReorderableState`, it
    composes with beam search.

    When ``cross_attention`` is enabled the decoder additionally attends to
    encoder memory. The memory is supplied through the ``initial_state`` (built by
    :class:`TransformerSequenceDecoderStateInitializer` from encoder outputs),
    making it usable as the decoder half of a Transformer sequence-to-sequence
    model with the same wiring as the LSTM decoder. The memory is projected into
    per-layer keys/values on the first step and cached in the state thereafter.

    Args:
        input_dim: Dimension of each input vector (``d_model``).
        num_heads: Number of attention heads. Must divide ``input_dim``.
        num_layers: Number of stacked decoder blocks.
        feedforward_dim: Hidden dimension of the position-wise feedforward network.
        dropout: Dropout rate applied to attention, feedforward, and residuals.
        cross_attention: Whether to add an encoder-memory cross-attention sublayer.
        memory_dim: Dimension of the encoder memory. Defaults to ``input_dim``.
            Only used when ``cross_attention`` is enabled.
        positional_encoder: Optional positional encoder to add position
            information, injected like :class:`TransformerEncoder`. During
            incremental decoding it is applied at the absolute positions implied
            by the cache length.
        activation: Feedforward activation module (default: ``torch.nn.ReLU()``).
        layer_norm_eps: Epsilon for layer normalization.
        norm_first: Whether to apply layer norm before each sublayer (pre-norm).
        attention_backend: Kernel used for scaled dot-product attention.
            ``"sdpa"`` (default) uses
            :func:`torch.nn.functional.scaled_dot_product_attention` and runs on
            any device (CPU, CUDA, MPS). ``"flex"`` uses
            :func:`~torch.nn.attention.flex_attention.flex_attention`, which fuses
            the offset-causal / padding mask into the kernel (best on CUDA under
            :func:`torch.compile`) but does not apply attention dropout and is
            only supported on CUDA/CPU/HPU devices -- notably **not** Apple MPS.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        cross_attention: bool = False,
        memory_dim: int | None = None,
        positional_encoder: BasePositionalEncoder | None = None,
        activation: nn.Module = nn.ReLU(),
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        attention_backend: AttentionBackend = "sdpa",
    ) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim {input_dim} must be divisible by num_heads {num_heads}")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")
        if attention_backend not in ("sdpa", "flex"):
            raise ValueError(f"Unsupported attention_backend {attention_backend!r}; expected 'sdpa' or 'flex'")
        if attention_backend == "flex" and not _FLEX_ATTENTION_AVAILABLE:
            raise ValueError("attention_backend='flex' requires torch.nn.attention.flex_attention")

        self._input_dim = input_dim
        self._cross_attention = cross_attention
        self._memory_dim = memory_dim if memory_dim is not None else input_dim
        self._positional_encoder = positional_encoder
        self._attention_backend = attention_backend

        self.layers = nn.ModuleList(
            _TransformerDecoderLayer(
                embed_dim=input_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                norm_first=norm_first,
                cross_attention=cross_attention,
                memory_dim=self._memory_dim,
                attention_backend=attention_backend,
            )
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def _decoder_layers(self) -> list["_TransformerDecoderLayer"]:
        """Typed view over ``self.layers`` (``nn.ModuleList`` erases the element type)."""
        return cast("list[_TransformerDecoderLayer]", list(self.layers))

    def _resolve_memory(
        self,
        initial_state: TransformerDecoderState | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Return per-layer cross keys/values and the source validity mask.

        Reuses the projections cached in ``initial_state`` when present, otherwise
        projects the encoder memory carried by the state once per layer. The
        returned validity mask marks non-padding source positions with ``True``.
        """
        if not self._cross_attention:
            return None, None, None
        if initial_state is None or (initial_state.cross_keys is None and initial_state.memory is None):
            raise ValueError(
                "Cross-attention decoder requires an initial_state carrying encoder memory "
                "(build it with TransformerSequenceDecoderStateInitializer)"
            )

        valid_mask = None if initial_state.memory_mask is None else initial_state.memory_mask.bool()
        if initial_state.cross_keys is not None:
            return initial_state.cross_keys, initial_state.cross_values, valid_mask

        memory = initial_state.memory
        assert memory is not None  # guaranteed by the guard above
        keys, values = [], []
        for layer in self._decoder_layers:
            key, value = layer.project_memory(memory)
            keys.append(key)
            values.append(value)
        return torch.stack(keys), torch.stack(values), valid_mask

    @staticmethod
    def _apply_positional_encoding(
        encoder: BasePositionalEncoder,
        inputs: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        """Add positional information at absolute positions ``offset .. offset + seq_len``.

        Positional encoders index positions from zero, so during incremental
        decoding (``offset > 0``) the encoder is run over a zero-padded prefix of
        length ``offset`` and only the trailing new positions are kept. This
        yields the correct absolute positions for additive (sinusoidal,
        learnable) and rotary encoders alike, matching the full-sequence pass.
        """
        if offset == 0:
            return encoder(inputs)
        batch_size, _, dim = inputs.shape
        prefix = inputs.new_zeros(batch_size, offset, dim)
        return encoder(torch.cat([prefix, inputs], dim=1))[:, offset:]

    def _build_self_mask(
        self,
        query_length: int,
        key_length: int,
        key_valid_mask: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "torch.Tensor | BlockMask":
        """Build the offset-causal self-attention mask for the active backend.

        ``key_valid_mask`` marks non-padding keys with ``True`` (or is ``None``).
        The same mask is shared by every layer, so it is built once per call.
        """
        if self._attention_backend == "flex":
            offset = key_length - query_length
            return create_block_mask(
                _TransformerDecoderLayer._causal_mask_mod(offset, key_valid_mask),
                B=batch_size,
                H=None,
                Q_LEN=query_length,
                KV_LEN=key_length,
                device=device,
            )
        key_padding_mask = None if key_valid_mask is None else ~key_valid_mask
        return _TransformerDecoderLayer._causal_padding_bias(query_length, key_length, key_padding_mask, dtype, device)

    def _build_cross_mask(
        self,
        query_length: int,
        key_valid_mask: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "torch.Tensor | BlockMask | None":
        """Build the cross-attention key-padding mask for the active backend."""
        if key_valid_mask is None:
            return None
        if self._attention_backend == "flex":
            return create_block_mask(
                _TransformerDecoderLayer._padding_mask_mod(key_valid_mask),
                B=batch_size,
                H=None,
                Q_LEN=query_length,
                KV_LEN=key_valid_mask.size(1),
                device=device,
            )
        return _TransformerDecoderLayer._padding_bias(~key_valid_mask, dtype)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        initial_state: TransformerDecoderState | None = None,
        params: None = None,
    ) -> tuple[torch.Tensor, TransformerDecoderState]:
        """Decode an input sequence, returning outputs and the updated cache."""
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch_size, seq_len, input_dim)")
        if inputs.size(-1) != self._input_dim:
            raise ValueError(f"Expected input dimension {self._input_dim}, got {inputs.size(-1)}")

        cached_self_keys = None if initial_state is None else initial_state.self_keys
        cached_self_values = None if initial_state is None else initial_state.self_values
        has_self_cache = cached_self_keys is not None
        offset = initial_state.length if initial_state is not None else 0

        hidden = inputs
        if self._positional_encoder is not None:
            hidden = self._apply_positional_encoding(self._positional_encoder, hidden, offset)
        hidden = self.dropout(hidden)

        cross_keys, cross_values, cross_valid_mask = self._resolve_memory(initial_state)

        batch_size = inputs.size(0)
        query_length = inputs.size(1)
        key_length = query_length + offset

        # A self-attention padding mask only aligns with the keys when the cache
        # is empty (the full-sequence pass); incremental steps append all-valid
        # tokens, so their padding mask no longer spans the cached keys.
        self_valid_mask = mask.bool() if (mask is not None and not has_self_cache) else None
        self_mask = self._build_self_mask(
            query_length, key_length, self_valid_mask, batch_size, hidden.dtype, inputs.device
        )
        cross_mask = self._build_cross_mask(query_length, cross_valid_mask, batch_size, hidden.dtype, inputs.device)

        new_keys, new_values = [], []
        for index, layer in enumerate(self._decoder_layers):
            past_key = None if cached_self_keys is None else cached_self_keys[index]
            past_value = None if cached_self_values is None else cached_self_values[index]
            memory_key = None if cross_keys is None else cross_keys[index]
            memory_value = None if cross_values is None else cross_values[index]
            hidden, key, value = layer(
                hidden,
                past_key,
                past_value,
                self_mask,
                memory_key,
                memory_value,
                cross_mask,
            )
            new_keys.append(key)
            new_values.append(value)

        if mask is not None:
            hidden = hidden * mask.unsqueeze(-1)

        state = TransformerDecoderState(
            self_keys=torch.stack(new_keys),
            self_values=torch.stack(new_values),
            # Raw memory is consumed once projected; keep the mask so later steps
            # can rebuild the cross-attention padding from the cached keys.
            memory=None,
            memory_mask=cross_valid_mask,
            cross_keys=cross_keys,
            cross_values=cross_values,
        )
        return hidden, state

    def get_input_dim(self) -> int:
        """Return the input vector dimension."""
        return self._input_dim

    def get_output_dim(self) -> int:
        """Return the output vector dimension."""
        return self._input_dim
