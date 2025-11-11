import jax.numpy as jnp

from formed.integrations.flax.modules.vectorizers import BagOfEmbeddingsSequenceVectorizer


class TestBagOfEmbeddingsSequenceVectorizer:
    def test_mean_pooling(self) -> None:
        """Test mean pooling without mask."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="mean")

        # (batch_size=2, seq_len=3, embedding_dim=4)
        inputs = jnp.array(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
            ]
        )

        output = vectorizer(inputs)

        assert output.shape == (2, 4)
        # First sample: mean of [1,2,3,4], [5,6,7,8], [9,10,11,12] = [5,6,7,8]
        assert jnp.allclose(output[0], jnp.array([5.0, 6.0, 7.0, 8.0]))
        # Second sample: mean of [1,1,1,1], [2,2,2,2], [3,3,3,3] = [2,2,2,2]
        assert jnp.allclose(output[1], jnp.array([2.0, 2.0, 2.0, 2.0]))

    def test_mean_pooling_with_mask(self) -> None:
        """Test mean pooling with mask to ignore padding."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="mean")

        inputs = jnp.array([[[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]], [[3.0, 3.0], [0.0, 0.0], [0.0, 0.0]]])
        mask = jnp.array([[True, True, False], [True, False, False]])

        output = vectorizer(inputs, mask=mask)

        assert output.shape == (2, 2)
        # First sample: mean of [1,1], [2,2] = [1.5, 1.5]
        assert jnp.allclose(output[0], jnp.array([1.5, 1.5]))
        # Second sample: only [3,3] = [3, 3]
        assert jnp.allclose(output[1], jnp.array([3.0, 3.0]))

    def test_max_pooling(self) -> None:
        """Test max pooling."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="max")

        inputs = jnp.array([[[1.0, 5.0], [2.0, 3.0], [4.0, 1.0]], [[3.0, 2.0], [1.0, 4.0], [2.0, 1.0]]])

        output = vectorizer(inputs)

        assert output.shape == (2, 2)
        # First sample: max of columns [4.0, 5.0]
        assert jnp.allclose(output[0], jnp.array([4.0, 5.0]))
        # Second sample: max of columns [3.0, 4.0]
        assert jnp.allclose(output[1], jnp.array([3.0, 4.0]))

    def test_min_pooling(self) -> None:
        """Test min pooling."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="min")

        inputs = jnp.array([[[1.0, 5.0], [2.0, 3.0], [4.0, 1.0]], [[3.0, 2.0], [1.0, 4.0], [2.0, 1.0]]])

        output = vectorizer(inputs)

        assert output.shape == (2, 2)
        # First sample: min of columns [1.0, 1.0]
        assert jnp.allclose(output[0], jnp.array([1.0, 1.0]))
        # Second sample: min of columns [1.0, 1.0]
        assert jnp.allclose(output[1], jnp.array([1.0, 1.0]))

    def test_sum_pooling(self) -> None:
        """Test sum pooling."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="sum")

        inputs = jnp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])

        output = vectorizer(inputs)

        assert output.shape == (2, 2)
        # First sample: sum [9.0, 12.0]
        assert jnp.allclose(output[0], jnp.array([9.0, 12.0]))
        # Second sample: sum [3.0, 3.0]
        assert jnp.allclose(output[1], jnp.array([3.0, 3.0]))

    def test_first_pooling(self) -> None:
        """Test first token pooling."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="first")

        inputs = jnp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])

        output = vectorizer(inputs)

        assert output.shape == (2, 2)
        # First sample: first token [1.0, 2.0]
        assert jnp.allclose(output[0], jnp.array([1.0, 2.0]))
        # Second sample: first token [7.0, 8.0]
        assert jnp.allclose(output[1], jnp.array([7.0, 8.0]))

    def test_last_pooling(self) -> None:
        """Test last token pooling."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="last")

        inputs = jnp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])

        output = vectorizer(inputs)

        assert output.shape == (2, 2)
        # Last token should be selected
        assert jnp.allclose(output[0], jnp.array([5.0, 6.0]))
        assert jnp.allclose(output[1], jnp.array([11.0, 12.0]))

    def test_last_pooling_with_mask(self) -> None:
        """Test last token pooling with mask to get actual last non-padding token."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="last")

        inputs = jnp.array([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], [[7.0, 8.0], [0.0, 0.0], [0.0, 0.0]]])
        mask = jnp.array([[True, True, False], [True, False, False]])

        output = vectorizer(inputs, mask=mask)

        assert output.shape == (2, 2)
        # First sample: last valid token [3.0, 4.0]
        assert jnp.allclose(output[0], jnp.array([3.0, 4.0]))
        # Second sample: last valid token [7.0, 8.0]
        assert jnp.allclose(output[1], jnp.array([7.0, 8.0]))

    def test_hier_pooling(self) -> None:
        """Test hierarchical pooling."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="hier", window_size=2)

        inputs = jnp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])

        output = vectorizer(inputs)

        assert output.shape == (1, 2)

    def test_with_normalization(self) -> None:
        """Test pooling with L2 normalization."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="mean", normalize=True)

        inputs = jnp.array([[[3.0, 4.0], [6.0, 8.0], [9.0, 12.0]]])

        output = vectorizer(inputs)

        assert output.shape == (1, 2)
        # After normalization, L2 norm should be 1
        norm = jnp.sqrt(jnp.sum(output**2, axis=-1))
        assert jnp.allclose(norm, 1.0, atol=1e-5)

    def test_dimension_agnostic(self) -> None:
        """Test that vectorizer is dimension-agnostic."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="mean")

        assert vectorizer.get_input_dim() is None
        assert callable(vectorizer.get_output_dim())

        # Test with different embedding dimensions
        inputs_64 = jnp.ones((2, 10, 64))
        output_64 = vectorizer(inputs_64)
        assert output_64.shape == (2, 64)

        inputs_128 = jnp.ones((2, 10, 128))
        output_128 = vectorizer(inputs_128)
        assert output_128.shape == (2, 128)

    def test_empty_mask_handling(self) -> None:
        """Test handling of sequences where all positions are masked."""
        vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="mean")

        inputs = jnp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = jnp.array([[False, False, False]])

        output = vectorizer(inputs, mask=mask)

        assert output.shape == (1, 2)
        # With all masked, should handle gracefully (likely zeros or similar)
        assert not jnp.any(jnp.isnan(output))
