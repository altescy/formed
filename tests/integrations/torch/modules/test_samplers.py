"""Tests for samplers module."""

import torch

from formed.integrations.torch.modules.samplers import (
    ArgmaxLabelSampler,
    BernoulliMultilabelSampler,
    MultinomialLabelSampler,
    ThresholdMultilabelSampler,
    TopKMultilabelSampler,
)


class TestArgmaxLabelSampler:
    def test_basic_sampling(self):
        """Test argmax label sampling with 2D logits."""
        sampler = ArgmaxLabelSampler()
        logits = torch.tensor([[1.0, 3.0, 2.0], [0.5, 0.2, 0.8], [2.0, 1.5, 3.5]])

        labels = sampler(logits)

        assert labels.shape == (3,)
        assert torch.equal(labels, torch.tensor([1, 2, 2]))

    def test_deterministic(self):
        """Test that argmax sampling is deterministic."""
        sampler = ArgmaxLabelSampler()
        logits = torch.randn(10, 5)

        labels1 = sampler(logits)
        labels2 = sampler(logits)

        assert torch.equal(labels1, labels2)

    def test_3d_logits(self):
        """Test argmax sampling with 3D logits (e.g., sequence labeling)."""
        sampler = ArgmaxLabelSampler()
        batch_size, seq_len, num_classes = 4, 8, 10
        logits = torch.randn(batch_size, seq_len, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size, seq_len)
        # Verify each label is in valid range
        assert torch.all(labels >= 0)
        assert torch.all(labels < num_classes)

    def test_single_sample(self):
        """Test argmax sampling with single sample."""
        sampler = ArgmaxLabelSampler()
        logits = torch.tensor([[1.0, 5.0, 2.0]])

        labels = sampler(logits)

        assert labels.shape == (1,)
        assert labels.item() == 1

    def test_ties(self):
        """Test argmax behavior with tied logits."""
        sampler = ArgmaxLabelSampler()
        # PyTorch argmax returns the first occurrence
        logits = torch.tensor([[2.0, 2.0, 1.0], [1.0, 3.0, 3.0]])

        labels = sampler(logits)

        assert labels.shape == (2,)
        assert labels[0].item() == 0  # First occurrence of max
        assert labels[1].item() == 1  # First occurrence of max


class TestMultinomialLabelSampler:
    def test_basic_sampling(self):
        """Test multinomial label sampling."""
        sampler = MultinomialLabelSampler()
        batch_size, num_classes = 20, 5
        logits = torch.randn(batch_size, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size,)
        assert torch.all(labels >= 0)
        assert torch.all(labels < num_classes)

    def test_stochastic(self):
        """Test that multinomial sampling is stochastic."""
        sampler = MultinomialLabelSampler()
        logits = torch.randn(100, 10)

        labels1 = sampler(logits)
        labels2 = sampler(logits)

        # With high probability, at least some labels should differ
        assert not torch.equal(labels1, labels2)

    def test_temperature_low(self):
        """Test multinomial sampling with low temperature (more deterministic)."""
        sampler = MultinomialLabelSampler()
        # Logits with clear winner
        logits = torch.tensor([[10.0, 1.0, 1.0]] * 50)

        labels = sampler(logits, params={"temperature": 0.1})

        # With low temperature, should mostly pick the highest logit
        assert (labels == 0).float().mean() > 0.8

    def test_temperature_high(self):
        """Test multinomial sampling with high temperature (more random)."""
        sampler = MultinomialLabelSampler()
        # Logits with clear winner
        logits = torch.tensor([[10.0, 1.0, 1.0]] * 100)

        labels = sampler(logits, params={"temperature": 10.0})

        # With high temperature, distribution should be more uniform
        # Not all samples should be 0
        unique_labels = torch.unique(labels)
        assert len(unique_labels) > 1

    def test_temperature_default(self):
        """Test multinomial sampling with default temperature."""
        sampler = MultinomialLabelSampler()
        logits = torch.randn(10, 5)

        labels = sampler(logits)

        assert labels.shape == (10,)
        assert torch.all(labels >= 0)
        assert torch.all(labels < 5)

    def test_3d_logits(self):
        """Test multinomial sampling with 3D logits."""
        sampler = MultinomialLabelSampler()
        batch_size, seq_len, num_classes = 4, 8, 10
        logits = torch.randn(batch_size, seq_len, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size, seq_len)
        assert torch.all(labels >= 0)
        assert torch.all(labels < num_classes)


class TestThresholdMultilabelSampler:
    def test_default_threshold(self):
        """Test threshold multilabel sampling with default threshold."""
        sampler = ThresholdMultilabelSampler()
        # Create logits that will produce probs around 0.5
        logits = torch.tensor([[0.0, 5.0, -5.0], [2.0, -2.0, 0.5]])

        labels = sampler(logits)

        assert labels.shape == (2, 3)
        # Check that outputs are binary
        assert torch.all((labels == 0.0) | (labels == 1.0))

    def test_custom_threshold(self):
        """Test threshold multilabel sampling with custom threshold."""
        sampler = ThresholdMultilabelSampler(threshold=0.7)
        # Create logits that map to known probabilities
        logits = torch.tensor([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]])

        labels = sampler(logits)

        assert labels.shape == (2, 3)
        # Verify threshold behavior
        probs = torch.sigmoid(logits)
        expected = (probs >= 0.7).float()
        assert torch.equal(labels, expected)

    def test_all_above_threshold(self):
        """Test when all logits are above threshold."""
        sampler = ThresholdMultilabelSampler(threshold=0.3)
        logits = torch.tensor([[5.0, 5.0, 5.0]])

        labels = sampler(logits)

        assert torch.all(labels == 1.0)

    def test_all_below_threshold(self):
        """Test when all logits are below threshold."""
        sampler = ThresholdMultilabelSampler(threshold=0.8)
        logits = torch.tensor([[-5.0, -5.0, -5.0]])

        labels = sampler(logits)

        assert torch.all(labels == 0.0)

    def test_deterministic(self):
        """Test that threshold sampling is deterministic."""
        sampler = ThresholdMultilabelSampler(threshold=0.5)
        logits = torch.randn(10, 8)

        labels1 = sampler(logits)
        labels2 = sampler(logits)

        assert torch.equal(labels1, labels2)


class TestTopKMultilabelSampler:
    def test_k_equals_1(self):
        """Test top-k sampling with k=1."""
        sampler = TopKMultilabelSampler(k=1)
        logits = torch.tensor([[1.0, 3.0, 2.0], [0.5, 0.2, 0.8]])

        labels = sampler(logits)

        assert labels.shape == (2, 3)
        # Each row should have exactly 1 label
        assert torch.equal(labels.sum(dim=-1), torch.tensor([1.0, 1.0]))
        # Check which labels are selected
        assert torch.equal(labels[0], torch.tensor([0.0, 1.0, 0.0]))
        assert torch.equal(labels[1], torch.tensor([0.0, 0.0, 1.0]))

    def test_k_equals_3(self):
        """Test top-k sampling with k=3."""
        sampler = TopKMultilabelSampler(k=3)
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])

        labels = sampler(logits)

        assert labels.shape == (1, 5)
        # Should have exactly 3 labels
        assert labels.sum().item() == 3.0
        # Top 3 should be indices 1, 4, 2 (logits 5.0, 4.0, 3.0)
        expected = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]])
        assert torch.equal(labels, expected)

    def test_k_equals_num_classes(self):
        """Test top-k when k equals number of classes."""
        sampler = TopKMultilabelSampler(k=3)
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        labels = sampler(logits)

        # Should select all available classes
        assert labels.sum().item() == 3.0
        assert torch.all(labels == 1.0)

    def test_deterministic(self):
        """Test that top-k sampling is deterministic."""
        sampler = TopKMultilabelSampler(k=2)
        logits = torch.randn(10, 8)

        labels1 = sampler(logits)
        labels2 = sampler(logits)

        assert torch.equal(labels1, labels2)

    def test_ties(self):
        """Test top-k behavior with tied logits."""
        sampler = TopKMultilabelSampler(k=2)
        logits = torch.tensor([[2.0, 2.0, 1.0, 2.0]])

        labels = sampler(logits)

        # Should select exactly k=2 labels
        assert labels.sum().item() == 2.0

    def test_batch_dimension(self):
        """Test top-k with batch dimension."""
        sampler = TopKMultilabelSampler(k=2)
        batch_size = 5
        num_classes = 10
        logits = torch.randn(batch_size, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size, num_classes)
        # Each sample should have exactly k labels
        assert torch.all(labels.sum(dim=-1) == 2.0)


class TestBernoulliMultilabelSampler:
    def test_basic_sampling(self):
        """Test Bernoulli multilabel sampling."""
        sampler = BernoulliMultilabelSampler()
        logits = torch.tensor([[0.0, 5.0, -5.0], [2.0, -2.0, 0.5]])

        labels = sampler(logits)

        assert labels.shape == (2, 3)
        # Check that outputs are binary
        assert torch.all((labels == 0.0) | (labels == 1.0))

    def test_stochastic(self):
        """Test that Bernoulli sampling is stochastic."""
        sampler = BernoulliMultilabelSampler()
        logits = torch.zeros(50, 10)  # Equal probability for all

        labels1 = sampler(logits)
        labels2 = sampler(logits)

        # With high probability, at least some labels should differ
        assert not torch.equal(labels1, labels2)

    def test_extreme_logits(self):
        """Test Bernoulli sampling with extreme logits."""
        sampler = BernoulliMultilabelSampler()

        # Very high logits (prob ≈ 1)
        high_logits = torch.tensor([[10.0, 10.0, 10.0]] * 10)
        labels_high = sampler(high_logits)
        # Most should be 1
        assert labels_high.float().mean() > 0.9

        # Very low logits (prob ≈ 0)
        low_logits = torch.tensor([[-10.0, -10.0, -10.0]] * 10)
        labels_low = sampler(low_logits)
        # Most should be 0
        assert labels_low.float().mean() < 0.1

    def test_independent_sampling(self):
        """Test that Bernoulli sampling is independent per class."""
        sampler = BernoulliMultilabelSampler()
        # Different probabilities for each class
        logits = torch.tensor([[5.0, -5.0, 0.0]] * 100)

        labels = sampler(logits)

        # First class should be mostly 1
        assert labels[:, 0].float().mean() > 0.9
        # Second class should be mostly 0
        assert labels[:, 1].float().mean() < 0.1
        # Third class should be around 0.5
        assert 0.3 < labels[:, 2].float().mean() < 0.7

    def test_batch_dimension(self):
        """Test Bernoulli sampling with batch dimension."""
        sampler = BernoulliMultilabelSampler()
        batch_size = 8
        num_classes = 12
        logits = torch.randn(batch_size, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size, num_classes)
        assert torch.all((labels == 0.0) | (labels == 1.0))

    def test_3d_logits(self):
        """Test Bernoulli sampling with 3D logits."""
        sampler = BernoulliMultilabelSampler()
        batch_size, seq_len, num_classes = 4, 8, 10
        logits = torch.randn(batch_size, seq_len, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size, seq_len, num_classes)
        assert torch.all((labels == 0.0) | (labels == 1.0))
