import torch

from formed.integrations.torch.modules.masks import (
    BaseAttentionMask,
    CausalMask,
    CombinedMask,
    SlidingWindowAttentionMask,
)


class TestAttentionMasks:
    """Tests for attention mask masks."""

    def test_causal_mask(self):
        """Test CausalMask creates lower triangular mask."""
        generator = CausalMask()
        mask = generator(seq_len=4, batch_size=1, device=torch.device("cpu"))

        assert mask.shape == (4, 4)
        assert mask.dtype == torch.float32

        # Check causal structure: mask[i, j] should be 0 if j <= i, -inf if j > i
        for i in range(4):
            for j in range(4):
                if j <= i:
                    assert mask[i, j] == 0.0
                else:
                    assert torch.isinf(mask[i, j]) and mask[i, j] < 0

    def test_combined_mask(self):
        """Test CombinedMask combines multiple causal masks."""
        causal1 = CausalMask()
        causal2 = CausalMask()
        combined = CombinedMask(masks=[causal1, causal2])

        mask = combined(seq_len=4, batch_size=2, device=torch.device("cpu"))

        assert mask is not None
        assert mask.shape == (4, 4)
        assert mask.dtype == torch.float32

        # Check causal structure: 0.0 for can attend, -inf for cannot attend
        for i in range(4):
            for j in range(4):
                if j <= i:
                    assert mask[i, j] == 0.0
                else:
                    assert torch.isinf(mask[i, j]) and mask[i, j] < 0

    def test_combined_mask_empty_returns_none(self):
        """Test CombinedMask returns None when no masks are provided or all return None."""
        combined = CombinedMask(masks=[])

        mask = combined(seq_len=4, batch_size=2, device=torch.device("cpu"), padding_mask=None)

        assert mask is None

    def test_sliding_window_mask(self):
        """Test SlidingWindowAttentionMask creates correct window pattern."""
        generator = SlidingWindowAttentionMask(window_size=1)
        mask = generator(seq_len=5, batch_size=1, device=torch.device("cpu"))

        assert mask.shape == (5, 5)
        assert mask.dtype == torch.float32

        # Check sliding window structure
        # Position i can attend to positions [i-window_size, i+window_size]
        expected = torch.tensor(
            [
                [0.0, 0.0, float("-inf"), float("-inf"), float("-inf")],  # pos 0: [0, 1]
                [0.0, 0.0, 0.0, float("-inf"), float("-inf")],  # pos 1: [0, 1, 2]
                [float("-inf"), 0.0, 0.0, 0.0, float("-inf")],  # pos 2: [1, 2, 3]
                [float("-inf"), float("-inf"), 0.0, 0.0, 0.0],  # pos 3: [2, 3, 4]
                [float("-inf"), float("-inf"), float("-inf"), 0.0, 0.0],  # pos 4: [3, 4]
            ]
        )

        assert torch.equal(mask, expected)

    def test_sliding_window_mask_larger_window(self):
        """Test SlidingWindowAttentionMask with larger window size."""
        generator = SlidingWindowAttentionMask(window_size=2)
        mask = generator(seq_len=5, batch_size=1, device=torch.device("cpu"))

        assert mask.shape == (5, 5)

        # With window_size=2, position 2 can attend to [0, 1, 2, 3, 4]
        assert mask[2, 0] == 0.0  # distance = 2, within window
        assert mask[2, 1] == 0.0  # distance = 1, within window
        assert mask[2, 2] == 0.0  # distance = 0, within window
        assert mask[2, 3] == 0.0  # distance = 1, within window
        assert mask[2, 4] == 0.0  # distance = 2, within window

        # Position 0 cannot attend to position 3 (distance = 3 > window_size)
        assert torch.isinf(mask[0, 3]) and mask[0, 3] < 0

    def test_sliding_window_mask_zero_window(self):
        """Test SlidingWindowAttentionMask with window_size=0 (self-attention only)."""
        generator = SlidingWindowAttentionMask(window_size=0)
        mask = generator(seq_len=4, batch_size=1, device=torch.device("cpu"))

        # With window_size=0, each position can only attend to itself
        for i in range(4):
            for j in range(4):
                if i == j:
                    assert mask[i, j] == 0.0
                else:
                    assert torch.isinf(mask[i, j]) and mask[i, j] < 0

    def test_sliding_window_mask_invalid_window_size(self):
        """Test that negative window_size raises error."""
        import pytest

        with pytest.raises(ValueError, match="window_size must be non-negative"):
            SlidingWindowAttentionMask(window_size=-1)

    def test_mask_protocol(self):
        """Test that mask generators implement BaseAttentionMask protocol."""
        assert isinstance(CausalMask(), BaseAttentionMask)
        assert isinstance(CombinedMask(masks=[]), BaseAttentionMask)
        assert isinstance(SlidingWindowAttentionMask(window_size=1), BaseAttentionMask)
