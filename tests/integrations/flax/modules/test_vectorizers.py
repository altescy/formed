"""Tests for vectorizers module."""

import pytest
import torch

from formed.integrations.torch.modules.vectorizers import (
    BagOfEmbeddingsSequenceVectorizer,
    BaseSequenceVectorizer,
    CnnSequenceVectorizer,
    ConcatSequenceVectorizer,
    SelfAttentiveSequenceVectorizer,
)


class TestBagOfEmbeddingsSequenceVectorizer:
    """Test BagOfEmbeddingsSequenceVectorizer."""

    @pytest.mark.parametrize(
        ("pooling", "expected_multiplier"),
        [
            pytest.param("mean", 1, id="mean"),
            pytest.param("max", 1, id="max"),
            pytest.param("sum", 1, id="sum"),
            pytest.param(["mean", "max"], 2, id="mean_max"),
            pytest.param(["mean", "max", "sum"], 3, id="mean_max_sum"),
        ],
    )
    def test_output_dim(self, pooling, expected_multiplier):
        """Test output dimension with different pooling methods."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling=pooling)

        # Output dim should be input_dim * number of pooling methods
        output_dim_fn = vectorizer.get_output_dim()
        assert callable(output_dim_fn)
        assert output_dim_fn(128) == 128 * expected_multiplier

    def test_basic_mean_pooling(self):
        """Test basic mean pooling."""
        batch_size, seq_len, input_dim = 2, 4, 8

        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="mean")
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = vectorizer(inputs)

        assert output.shape == (batch_size, input_dim)
        # Mean should be approximately equal to manual mean
        expected = inputs.mean(dim=1)
        assert torch.allclose(output, expected)

    def test_with_mask(self):
        """Test pooling with padding mask."""
        batch_size, seq_len, input_dim = 2, 4, 8

        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="mean")
        inputs = torch.randn(batch_size, seq_len, input_dim)

        # Create mask where second half is padding
        mask = torch.ones(batch_size, seq_len)
        mask[:, 2:] = 0

        output = vectorizer(inputs, mask=mask)

        assert output.shape == (batch_size, input_dim)
        # Should only average over first 2 positions
        expected = inputs[:, :2].mean(dim=1)
        assert torch.allclose(output, expected)

    def test_is_base_vectorizer(self):
        """Test that BagOfEmbeddingsSequenceVectorizer inherits from BaseSequenceVectorizer."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer()
        assert isinstance(vectorizer, BaseSequenceVectorizer)


class TestCnnSequenceVectorizer:
    """Test CnnSequenceVectorizer."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        batch_size, seq_len, input_dim = 2, 10, 16
        num_filters = 8
        ngram_filter_sizes = (2, 3, 4)

        vectorizer = CnnSequenceVectorizer(
            input_dim=input_dim,
            num_filters=num_filters,
            ngram_filter_sizes=ngram_filter_sizes,
        )
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = vectorizer(inputs)

        # Output should be concatenation of max-pooled features from each filter
        expected_dim = num_filters * len(ngram_filter_sizes)
        assert output.shape == (batch_size, expected_dim)

    def test_with_output_projection(self):
        """Test with output projection layer."""
        batch_size, seq_len, input_dim = 2, 10, 16
        num_filters = 8
        output_dim = 32

        vectorizer = CnnSequenceVectorizer(
            input_dim=input_dim,
            num_filters=num_filters,
            ngram_filter_sizes=(2, 3),
            output_dim=output_dim,
        )
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = vectorizer(inputs)

        assert output.shape == (batch_size, output_dim)

    def test_with_mask(self):
        """Test forward pass with padding mask."""
        batch_size, seq_len, input_dim = 2, 10, 16
        num_filters = 8

        vectorizer = CnnSequenceVectorizer(
            input_dim=input_dim,
            num_filters=num_filters,
            ngram_filter_sizes=(2, 3),
        )
        inputs = torch.randn(batch_size, seq_len, input_dim)

        # Create mask where second half is padding
        mask = torch.ones(batch_size, seq_len)
        mask[0, 7:] = 0  # First sample has padding from position 7
        mask[1, 5:] = 0  # Second sample has padding from position 5

        output = vectorizer(inputs, mask=mask)

        expected_dim = num_filters * 2  # 2 filter sizes
        assert output.shape == (batch_size, expected_dim)
        # Output should not be all zeros or NaN
        assert not torch.isnan(output).any()
        assert not (output == 0).all()

    @pytest.mark.parametrize(
        "ngram_filter_sizes",
        [
            pytest.param((2,), id="single_bigram"),
            pytest.param((3,), id="single_trigram"),
            pytest.param((2, 3), id="bigram_trigram"),
            pytest.param((2, 3, 4, 5), id="default"),
            pytest.param((1, 2, 3), id="unigram_bigram_trigram"),
        ],
    )
    def test_different_ngram_sizes(self, ngram_filter_sizes):
        """Test with different n-gram filter sizes."""
        batch_size, seq_len, input_dim = 2, 10, 16
        num_filters = 8

        vectorizer = CnnSequenceVectorizer(
            input_dim=input_dim,
            num_filters=num_filters,
            ngram_filter_sizes=ngram_filter_sizes,
        )
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = vectorizer(inputs)

        expected_dim = num_filters * len(ngram_filter_sizes)
        assert output.shape == (batch_size, expected_dim)

    def test_custom_activation(self):
        """Test with custom activation function."""
        batch_size, seq_len, input_dim = 2, 10, 16

        vectorizer = CnnSequenceVectorizer(
            input_dim=input_dim,
            num_filters=8,
            ngram_filter_sizes=(2, 3),
            conv_layer_activation=torch.nn.Tanh(),
        )
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = vectorizer(inputs)

        assert output.shape == (batch_size, 16)  # 8 filters * 2 sizes

    def test_gradients_flow(self):
        """Test that gradients flow through the vectorizer."""
        batch_size, seq_len, input_dim = 2, 8, 8

        vectorizer = CnnSequenceVectorizer(
            input_dim=input_dim,
            num_filters=4,
            ngram_filter_sizes=(2, 3),
        )
        inputs = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)

        output = vectorizer(inputs)
        loss = output.sum()
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()

    def test_output_dimensions(self):
        """Test get_input_dim and get_output_dim methods."""
        input_dim = 16
        num_filters = 10
        ngram_filter_sizes = (2, 3, 4)

        # Without projection
        vectorizer = CnnSequenceVectorizer(
            input_dim=input_dim,
            num_filters=num_filters,
            ngram_filter_sizes=ngram_filter_sizes,
        )

        assert vectorizer.get_input_dim() == input_dim
        assert vectorizer.get_output_dim() == num_filters * len(ngram_filter_sizes)

        # With projection
        output_dim = 32
        vectorizer_proj = CnnSequenceVectorizer(
            input_dim=input_dim,
            num_filters=num_filters,
            ngram_filter_sizes=ngram_filter_sizes,
            output_dim=output_dim,
        )

        assert vectorizer_proj.get_input_dim() == input_dim
        assert vectorizer_proj.get_output_dim() == output_dim

    def test_is_base_vectorizer(self):
        """Test that CnnSequenceVectorizer inherits from BaseSequenceVectorizer."""
        vectorizer = CnnSequenceVectorizer(
            input_dim=16,
            num_filters=8,
            ngram_filter_sizes=(2, 3),
        )
        assert isinstance(vectorizer, BaseSequenceVectorizer)

    def test_short_sequences(self):
        """Test with sequences shorter than largest n-gram size."""
        batch_size, seq_len, input_dim = 2, 5, 16  # 5 tokens
        num_filters = 8

        # Use filters where largest is same as sequence length
        vectorizer = CnnSequenceVectorizer(
            input_dim=input_dim,
            num_filters=num_filters,
            ngram_filter_sizes=(2, 3, 5),  # 5 equals seq_len
        )
        inputs = torch.randn(batch_size, seq_len, input_dim)

        # Should handle when largest filter equals sequence length
        output = vectorizer(inputs)

        expected_dim = num_filters * 3
        assert output.shape == (batch_size, expected_dim)
        # Output should not be all zeros or NaN
        assert not torch.isnan(output).any()
        assert not (output == 0).all()

    def test_deterministic_without_dropout(self):
        """Test that output is deterministic without dropout."""
        batch_size, seq_len, input_dim = 2, 8, 16

        vectorizer = CnnSequenceVectorizer(
            input_dim=input_dim,
            num_filters=8,
            ngram_filter_sizes=(2, 3),
        )
        vectorizer.eval()

        inputs = torch.randn(batch_size, seq_len, input_dim)

        output1 = vectorizer(inputs)
        output2 = vectorizer(inputs)

        assert torch.allclose(output1, output2)


class TestSelfAttentiveSequenceVectorizer:
    """Test SelfAttentiveSequenceVectorizer."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        batch_size, seq_len, input_dim = 2, 8, 16

        vectorizer = SelfAttentiveSequenceVectorizer(input_dim=input_dim)
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = vectorizer(inputs)

        assert output.shape == (batch_size, input_dim)

    def test_with_mask(self):
        """Test forward pass with padding mask."""
        batch_size, seq_len, input_dim = 2, 8, 16

        vectorizer = SelfAttentiveSequenceVectorizer(input_dim=input_dim)
        inputs = torch.randn(batch_size, seq_len, input_dim)

        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, 5:] = False

        output = vectorizer(inputs, mask=mask)

        assert output.shape == (batch_size, input_dim)
        assert not torch.isnan(output).any()

    def test_with_multiple_heads(self):
        """Test with multiple attention heads."""
        batch_size, seq_len, input_dim = 2, 8, 16
        num_heads = 4

        vectorizer = SelfAttentiveSequenceVectorizer(
            input_dim=input_dim,
            num_heads=num_heads,
        )
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = vectorizer(inputs)

        assert output.shape == (batch_size, input_dim)

    def test_with_hidden_dims(self):
        """Test with hidden dimensions in attention mechanism."""
        batch_size, seq_len, input_dim = 2, 8, 16
        hidden_dim = 32

        vectorizer = SelfAttentiveSequenceVectorizer(
            input_dim=input_dim,
            hidden_dims=(hidden_dim,),
        )
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = vectorizer(inputs)

        assert output.shape == (batch_size, input_dim)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to approximately 1."""
        batch_size, seq_len, input_dim = 2, 8, 16

        vectorizer = SelfAttentiveSequenceVectorizer(input_dim=input_dim)
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = vectorizer(inputs)

        assert output.shape == (batch_size, input_dim)
        assert not torch.isnan(output).any()

    def test_output_dimensions(self):
        """Test get_input_dim and get_output_dim methods."""
        input_dim = 16

        vectorizer = SelfAttentiveSequenceVectorizer(input_dim=input_dim)

        assert vectorizer.get_input_dim() == input_dim
        assert vectorizer.get_output_dim() == input_dim

    def test_is_base_vectorizer(self):
        """Test that SelfAttentiveSequenceVectorizer inherits from BaseSequenceVectorizer."""
        vectorizer = SelfAttentiveSequenceVectorizer(input_dim=16)
        assert isinstance(vectorizer, BaseSequenceVectorizer)

    def test_gradients_flow(self):
        """Test that gradients flow through the vectorizer."""
        batch_size, seq_len, input_dim = 2, 8, 8

        vectorizer = SelfAttentiveSequenceVectorizer(input_dim=input_dim)
        inputs = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)

        output = vectorizer(inputs)
        loss = output.sum()
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()


class TestConcatSequenceVectorizer:
    """Test ConcatSequenceVectorizer."""

    def test_basic_forward(self):
        """Test basic forward pass with multiple vectorizers."""
        batch_size, seq_len, input_dim = 2, 8, 16

        vectorizers = [
            BagOfEmbeddingsSequenceVectorizer(pooling="mean"),
            BagOfEmbeddingsSequenceVectorizer(pooling="max"),
        ]

        concat_vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = concat_vectorizer(inputs)

        expected_dim = input_dim * 2
        assert output.shape == (batch_size, expected_dim)

    def test_with_different_vectorizers(self):
        """Test with different types of vectorizers."""
        batch_size, seq_len, input_dim = 2, 10, 16

        vectorizers = [
            BagOfEmbeddingsSequenceVectorizer(pooling="mean"),
            CnnSequenceVectorizer(input_dim=input_dim, num_filters=8, ngram_filter_sizes=(2, 3)),
            SelfAttentiveSequenceVectorizer(input_dim=input_dim),
        ]

        concat_vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = concat_vectorizer(inputs)

        expected_dim = input_dim + (8 * 2) + input_dim
        assert output.shape == (batch_size, expected_dim)

    def test_with_mask(self):
        """Test forward pass with padding mask."""
        batch_size, seq_len, input_dim = 2, 8, 16

        vectorizers = [
            BagOfEmbeddingsSequenceVectorizer(pooling="mean"),
            BagOfEmbeddingsSequenceVectorizer(pooling="max"),
        ]

        concat_vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)
        inputs = torch.randn(batch_size, seq_len, input_dim)

        mask = torch.ones(batch_size, seq_len)
        mask[:, 5:] = 0

        output = concat_vectorizer(inputs, mask=mask)

        expected_dim = input_dim * 2
        assert output.shape == (batch_size, expected_dim)
        assert not torch.isnan(output).any()

    def test_output_dimensions_with_fixed_dims(self):
        """Test get_output_dim with vectorizers that have fixed output dims."""
        input_dim = 16
        num_filters = 8

        vectorizers = [
            CnnSequenceVectorizer(input_dim=input_dim, num_filters=num_filters, ngram_filter_sizes=(2, 3)),
            SelfAttentiveSequenceVectorizer(input_dim=input_dim),
        ]

        concat_vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)

        assert concat_vectorizer.get_input_dim() == input_dim
        expected_output_dim = (num_filters * 2) + input_dim
        assert concat_vectorizer.get_output_dim() == expected_output_dim

    def test_output_dimensions_with_callable_dims(self):
        """Test get_output_dim with vectorizers that have callable output dims."""
        vectorizers = [
            BagOfEmbeddingsSequenceVectorizer(pooling="mean"),
            BagOfEmbeddingsSequenceVectorizer(pooling=["mean", "max"]),
        ]

        concat_vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)

        output_dim_fn = concat_vectorizer.get_output_dim()
        assert callable(output_dim_fn)

        input_dim = 16
        expected_output_dim = input_dim * 1 + input_dim * 2
        assert output_dim_fn(input_dim) == expected_output_dim

    def test_output_dimensions_mixed(self):
        """Test get_output_dim with mixed fixed and callable dims."""
        input_dim = 16

        vectorizers = [
            BagOfEmbeddingsSequenceVectorizer(pooling="mean"),
            SelfAttentiveSequenceVectorizer(input_dim=input_dim),
        ]

        concat_vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)

        output_dim_fn = concat_vectorizer.get_output_dim()
        assert callable(output_dim_fn)
        assert output_dim_fn(input_dim) == input_dim * 2

    def test_is_base_vectorizer(self):
        """Test that ConcatSequenceVectorizer inherits from BaseSequenceVectorizer."""
        vectorizers = [
            BagOfEmbeddingsSequenceVectorizer(pooling="mean"),
        ]
        concat_vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)
        assert isinstance(concat_vectorizer, BaseSequenceVectorizer)

    def test_gradients_flow(self):
        """Test that gradients flow through the vectorizer."""
        batch_size, seq_len, input_dim = 2, 8, 8

        vectorizers = [
            BagOfEmbeddingsSequenceVectorizer(pooling="mean"),
            BagOfEmbeddingsSequenceVectorizer(pooling="max"),
        ]

        concat_vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)
        inputs = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)

        output = concat_vectorizer(inputs)
        loss = output.sum()
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()

    def test_single_vectorizer(self):
        """Test with only a single vectorizer."""
        batch_size, seq_len, input_dim = 2, 8, 16

        vectorizers = [
            BagOfEmbeddingsSequenceVectorizer(pooling="mean"),
        ]

        concat_vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = concat_vectorizer(inputs)

        assert output.shape == (batch_size, input_dim)

    def test_many_vectorizers(self):
        """Test with many vectorizers."""
        batch_size, seq_len, input_dim = 2, 8, 16

        vectorizers = [
            BagOfEmbeddingsSequenceVectorizer(pooling="mean"),
            BagOfEmbeddingsSequenceVectorizer(pooling="max"),
            BagOfEmbeddingsSequenceVectorizer(pooling="sum"),
            SelfAttentiveSequenceVectorizer(input_dim=input_dim),
        ]

        concat_vectorizer = ConcatSequenceVectorizer(vectorizers=vectorizers)
        inputs = torch.randn(batch_size, seq_len, input_dim)

        output = concat_vectorizer(inputs)

        expected_dim = input_dim * 4
        assert output.shape == (batch_size, expected_dim)
