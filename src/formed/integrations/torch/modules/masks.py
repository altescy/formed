"""Attention mask generation for transformer models.
This module provides reusable attention mask generators for transformer-based
models in PyTorch. Attention masks control which positions in a sequence
can attend to which other positions, enabling various attention patterns
such as causal masking for autoregressive models or sliding window attention
for long sequences.

Key Components:
    - BaseAttentionMask: Abstract base class for attention mask generators
    - CausalMask: Generates causal (autoregressive) attention masks
    - SlidingWindowAttentionMask: Generates sliding window attention masks
    - CombinedMask: Combines multiple attention masks into a single mask


Features:
    - Standardized attention mask format compatible with PyTorch Transformer modules
    - Support for batch-wise and sequence-wise masks
    - Easily extensible via registration system for custom masks

Examples:
    >>> from formed.integrations.torch.modules import CausalMask
    >>>
    >>> # Create a causal mask generator
    >>> mask_generator = CausalMask()
    >>>
    >>> # Generate a causal mask for sequence length 5 and batch size 2
    >>> mask = mask_generator(seq_len=5, batch_size=2, device=torch.device('cpu'))
    >>> # mask shape will be (5, 5) with float values: 0.0 for attendable positions,
    >>> # float('-inf') for masked positions

"""

import abc
from collections.abc import Sequence
from typing import Optional

import torch
from colt import Registrable


class BaseAttentionMask(Registrable, abc.ABC):
    """Base class for attention mask generation.

    Attention masks control which positions can attend to which other positions
    in transformer models.

    All attention masks must return a mask of shape (seq_len, seq_len) or
    (batch_size, seq_len, seq_len) using float values where:
    - 0.0 indicates positions that CAN be attended to
    - float('-inf') indicates positions that should NOT be attended to

    This standardized format ensures compatibility with PyTorch's
    TransformerEncoder.mask parameter.

    """

    @abc.abstractmethod
    def __call__(
        self,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Generate attention mask for self-attention.

        Args:
            seq_len: Sequence length.
            batch_size: Batch size.
            device: Device to create mask on.
            padding_mask: Optional padding mask of shape (batch_size, seq_len).
                         True/1 indicates valid positions, False/0 indicates padding.

        Returns:
            Attention mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len).
            None if no masking is needed.
            Uses float values: 0.0 for positions that can be attended to,
            float('-inf') for positions that should NOT be attended to.

        """
        ...


@BaseAttentionMask.register("causal")
class CausalMask(BaseAttentionMask):
    """Generates causal (autoregressive) attention masks.

    Causal masks ensure that each position can only attend to itself
    and previous positions, enabling autoregressive generation.

    Examples:
        >>> masks = CausalMask()
        >>> mask = masks(seq_len=4, batch_size=1, device=torch.device('cpu'))
        >>> # mask[i, j] = 0.0 if j <= i else float('-inf')

    """

    def __call__(
        self,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate causal attention mask.

        Args:
            seq_len: Sequence length.
            batch_size: Batch size.
            device: Device to create mask on.
            padding_mask: Optional padding mask (not used for causal masking).

        Returns:
            Causal mask of shape (seq_len, seq_len).

        """
        # Create causal mask: lower triangular matrix
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask


@BaseAttentionMask.register("sliding_window")
class SlidingWindowAttentionMask(BaseAttentionMask):
    """Sliding window attention mask.

    Restricts attention to a local window around each position, enabling
    efficient processing of long sequences. Commonly used in models like
    Longformer and Mistral.

    Args:
        window_size: Size of the attention window on each side.
                     Total window is (2 * window_size + 1) centered on each position.

    Examples:
        >>> # Window size of 1 means each position can attend to itself and
        >>> # one position on each side
        >>> mask_gen = SlidingWindowAttentionMask(window_size=1)
        >>> mask = mask_gen(seq_len=4, batch_size=1, device=torch.device('cpu'))
        >>> # Position 0: can attend to [0, 1]
        >>> # Position 1: can attend to [0, 1, 2]
        >>> # Position 2: can attend to [1, 2, 3]
        >>> # Position 3: can attend to [2, 3]

    """

    def __init__(self, window_size: int) -> None:
        if window_size < 0:
            raise ValueError(f"window_size must be non-negative, got {window_size}")
        self.window_size = window_size

    def __call__(
        self,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate sliding window attention mask.

        Args:
            seq_len: Sequence length.
            batch_size: Batch size.
            device: Device to create mask on.
            padding_mask: Optional padding mask (not used for sliding window).

        Returns:
            Attention mask of shape (seq_len, seq_len) where positions outside
            the window are masked with float('-inf').

        """
        # Create a matrix of positions
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        distances = torch.abs(positions - positions.T)

        # Mask positions outside the window
        mask = torch.zeros(seq_len, seq_len, device=device)
        mask = mask.masked_fill(distances > self.window_size, float("-inf"))

        return mask


@BaseAttentionMask.register("combined")
class CombinedMask(BaseAttentionMask):
    """Combines multiple attention masks.

    Applies multiple masks in sequence and combines their results.
    A position is masked if ANY mask blocks it (logical OR for -inf values).

    Args:
        masks: List of attention masks to combine.

    Examples:
        >>> # Combine multiple structural masks
        >>> mask1 = CausalMask()
        >>> mask2 = SomeOtherMask()
        >>> combined = CombinedMask(masks=[mask1, mask2])

    """

    def __init__(self, masks: Sequence[BaseAttentionMask]) -> None:
        self.masks = masks

    def __call__(
        self,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Generate combined attention mask.

        Args:
            seq_len: Sequence length.
            batch_size: Batch size.
            device: Device to create mask on.
            padding_mask: Optional padding mask.

        Returns:
            Combined attention mask of shape (seq_len, seq_len) or
            (batch_size, seq_len, seq_len), or None if all masks return None.

        """
        generated_masks = []
        for mask_generator in self.masks:
            mask = mask_generator(seq_len, batch_size, device, padding_mask)
            if mask is not None:
                generated_masks.append(mask)

        if not generated_masks:
            return None

        # All masks now have compatible shapes (either (seq_len, seq_len) or
        # (batch_size, seq_len, seq_len)), so we can combine them directly
        # using element-wise minimum (since -inf < 0.0)
        combined_mask = generated_masks[0]
        for mask in generated_masks[1:]:
            # Use minimum to combine: if either mask has -inf, result is -inf
            combined_mask = torch.minimum(combined_mask, mask)

        return combined_mask
