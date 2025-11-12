import jax.numpy as jnp

from formed.integrations.flax.modules.weighters import BalancedByDistributionLabelWeighter, StaticLabelWeighter


class TestStaticLabelWeighter:
    def test_basic_weighting(self) -> None:
        """Test basic static label weighting."""
        weights = jnp.array([1.0, 2.0, 3.0])
        weighter = StaticLabelWeighter(weights=weights)

        logits = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        labels = jnp.array([0, 1])

        output = weighter(logits, labels)

        # Output should have shape (batch_size, num_classes)
        assert output.shape == (1, 3)
        assert jnp.allclose(output[0], weights)

    def test_weight_preservation(self) -> None:
        """Test that weights are preserved correctly."""
        weights = jnp.array([0.5, 1.5, 2.5, 3.5])
        weighter = StaticLabelWeighter(weights=weights)

        logits = jnp.array([[0.0, 0.0, 0.0, 0.0]])
        labels = jnp.array([0])

        output = weighter(logits, labels)

        assert jnp.allclose(output[0], weights)

    def test_binary_classification(self) -> None:
        """Test static weighting for binary classification."""
        weights = jnp.array([1.0, 5.0])
        weighter = StaticLabelWeighter(weights=weights)

        logits = jnp.array([[0.1, 0.9], [0.8, 0.2]])
        labels = jnp.array([0, 1])

        output = weighter(logits, labels)

        assert output.shape == (1, 2)
        assert jnp.allclose(output[0], weights)

    def test_multiclass_classification(self) -> None:
        """Test static weighting for multiclass classification."""
        weights = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weighter = StaticLabelWeighter(weights=weights)

        logits = jnp.array([[0.2] * 5, [0.2] * 5, [0.2] * 5])
        labels = jnp.array([0, 2, 4])

        output = weighter(logits, labels)

        assert output.shape == (1, 5)
        assert jnp.allclose(output[0], weights)

    def test_weights_broadcast(self) -> None:
        """Test that weights broadcast correctly across batch dimension."""
        weights = jnp.array([1.0, 2.0, 3.0])
        weighter = StaticLabelWeighter(weights=weights)

        batch_size = 10
        logits = jnp.ones((batch_size, 3))
        labels = jnp.zeros(batch_size, dtype=jnp.int32)

        output = weighter(logits, labels)

        # Should broadcast to each sample in batch
        assert output.shape == (1, 3)

    def test_different_weight_values(self) -> None:
        """Test with different weight values including zero and large values."""
        weights = jnp.array([0.0, 0.1, 1.0, 10.0, 100.0])
        weighter = StaticLabelWeighter(weights=weights)

        logits = jnp.array([[0.5] * 5])
        labels = jnp.array([0])

        output = weighter(logits, labels)

        assert jnp.allclose(output[0], weights)


class TestBalancedByDistributionLabelWeighter:
    def test_basic_balancing(self) -> None:
        """Test basic balanced weighting."""
        distribution = jnp.array([0.5, 0.3, 0.2])
        weighter = BalancedByDistributionLabelWeighter(distribution=distribution)

        logits = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        labels = jnp.array([0, 1])

        output = weighter(logits, labels)

        # Output should have shape (batch_size, num_classes)
        assert output.shape == (1, 3)
        # Weights should be inversely proportional to class frequency
        assert output[0, 0] < output[0, 2]  # Common class gets lower weight

    def test_uniform_distribution(self) -> None:
        """Test with uniform class distribution."""
        distribution = jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        weighter = BalancedByDistributionLabelWeighter(distribution=distribution)

        logits = jnp.array([[0.0, 0.0, 0.0]])
        labels = jnp.array([0])

        output = weighter(logits, labels)

        # With uniform distribution, all weights should be equal
        assert jnp.allclose(output[0, 0], output[0, 1])
        assert jnp.allclose(output[0, 1], output[0, 2])

    def test_imbalanced_distribution(self) -> None:
        """Test with highly imbalanced class distribution."""
        # Class 0 is very rare, class 1 is very common
        distribution = jnp.array([0.01, 0.98, 0.01])
        weighter = BalancedByDistributionLabelWeighter(distribution=distribution)

        logits = jnp.array([[0.0, 0.0, 0.0]])
        labels = jnp.array([0])

        output = weighter(logits, labels)

        # Rare classes should get much higher weights
        assert output[0, 0] > output[0, 1]
        assert output[0, 2] > output[0, 1]
        # Class 0 and 2 (both rare) should have similar weights
        assert jnp.allclose(output[0, 0], output[0, 2], rtol=0.1)

    def test_binary_classification(self) -> None:
        """Test balanced weighting for binary classification."""
        distribution = jnp.array([0.7, 0.3])
        weighter = BalancedByDistributionLabelWeighter(distribution=distribution)

        logits = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        labels = jnp.array([0, 1])

        output = weighter(logits, labels)

        assert output.shape == (1, 2)
        # Less common class should get higher weight
        assert output[0, 1] > output[0, 0]

    def test_epsilon_prevents_division_by_zero(self) -> None:
        """Test that epsilon prevents division by zero."""
        # One class has zero probability
        distribution = jnp.array([0.5, 0.5, 0.0])
        weighter = BalancedByDistributionLabelWeighter(distribution=distribution, eps=1e-8)

        logits = jnp.array([[0.0, 0.0, 0.0]])
        labels = jnp.array([0])

        output = weighter(logits, labels)

        # Should not produce NaN or Inf
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_custom_epsilon(self) -> None:
        """Test with custom epsilon value."""
        distribution = jnp.array([0.5, 0.3, 0.2])
        weighter = BalancedByDistributionLabelWeighter(distribution=distribution, eps=1e-5)

        logits = jnp.array([[0.0, 0.0, 0.0]])
        labels = jnp.array([0])

        output = weighter(logits, labels)

        assert output.shape == (1, 3)
        assert not jnp.any(jnp.isnan(output))

    def test_multiclass_classification(self) -> None:
        """Test balanced weighting for multiclass classification."""
        distribution = jnp.array([0.1, 0.2, 0.3, 0.25, 0.15])
        weighter = BalancedByDistributionLabelWeighter(distribution=distribution)

        logits = jnp.array([[0.2] * 5])
        labels = jnp.array([0])

        output = weighter(logits, labels)

        assert output.shape == (1, 5)
        # Class 0 (rarest) should have highest weight
        assert output[0, 0] > output[0, 2]  # 0.1 vs 0.3
        # Class 2 (most common) should have lowest weight
        assert output[0, 2] < output[0, 0]

    def test_weights_sum_relationship(self) -> None:
        """Test that weights follow expected inverse relationship."""
        distribution = jnp.array([0.6, 0.3, 0.1])
        weighter = BalancedByDistributionLabelWeighter(distribution=distribution, eps=0.0)

        logits = jnp.array([[0.0, 0.0, 0.0]])
        labels = jnp.array([0])

        output = weighter(logits, labels)

        # Verify inverse relationship: weight ∝ 1/distribution
        num_classes = len(distribution)
        expected_weights = 1.0 / (distribution * num_classes)

        assert jnp.allclose(output[0], expected_weights)

    def test_params_ignored(self) -> None:
        """Test that params argument is ignored (for interface compatibility)."""
        distribution = jnp.array([0.5, 0.5])
        weighter = BalancedByDistributionLabelWeighter(distribution=distribution)

        logits = jnp.array([[0.0, 0.0]])
        labels = jnp.array([0])

        output_with_params = weighter(logits, labels, params=None)
        output_without_params = weighter(logits, labels)

        assert jnp.allclose(output_with_params, output_without_params)
