import jax.numpy as jnp

from formed.integrations.flax.modules.losses import CrossEntropyLoss
from formed.integrations.flax.modules.weighters import BalancedByDistributionLabelWeighter, StaticLabelWeighter


class TestCrossEntropyLoss:
    def test_basic_cross_entropy(self) -> None:
        """Test basic cross-entropy loss computation."""
        loss_fn = CrossEntropyLoss()

        logits = jnp.array([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]])
        labels = jnp.array([0, 1])

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        assert loss > 0

    def test_with_perfect_predictions(self) -> None:
        """Test cross-entropy with perfect predictions (should be near zero)."""
        loss_fn = CrossEntropyLoss()

        # Very high logits for correct class
        logits = jnp.array([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        labels = jnp.array([0, 1])

        loss = loss_fn(logits, labels)

        assert loss < 0.1  # Should be very small

    def test_with_wrong_predictions(self) -> None:
        """Test cross-entropy with completely wrong predictions."""
        loss_fn = CrossEntropyLoss()

        # High logits for wrong class
        logits = jnp.array([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        labels = jnp.array([1, 0])  # Wrong labels

        loss = loss_fn(logits, labels)

        assert loss > 5  # Should be high

    def test_mean_reduction(self) -> None:
        """Test cross-entropy with mean reduction."""
        loss_fn = CrossEntropyLoss(reduce="mean")

        logits = jnp.array([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0], [1.0, 0.1, 2.0]])
        labels = jnp.array([0, 1, 2])

        loss = loss_fn(logits, labels)

        assert loss.shape == ()

    def test_sum_reduction(self) -> None:
        """Test cross-entropy with sum reduction."""
        loss_fn = CrossEntropyLoss(reduce="sum")

        logits = jnp.array([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0], [1.0, 0.1, 2.0]])
        labels = jnp.array([0, 1, 2])

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        # Sum should be larger than mean for multiple samples
        assert loss > 0

    def test_with_static_weighter(self) -> None:
        """Test cross-entropy with static class weights."""
        # Higher weight for class 0
        weights = jnp.array([2.0, 1.0, 1.0])
        weighter = StaticLabelWeighter(weights=weights)
        loss_fn = CrossEntropyLoss(weighter=weighter)

        logits = jnp.array([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]])
        labels = jnp.array([0, 1])

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        assert loss > 0

    def test_with_balanced_weighter(self) -> None:
        """Test cross-entropy with balanced class weights."""
        # Class distribution: class 0 is rare, class 1 is common
        distribution = jnp.array([0.1, 0.5, 0.4])
        weighter = BalancedByDistributionLabelWeighter(distribution=distribution)
        loss_fn = CrossEntropyLoss(weighter=weighter)

        logits = jnp.array([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]])
        labels = jnp.array([0, 1])

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        assert loss > 0

    def test_binary_classification(self) -> None:
        """Test cross-entropy for binary classification."""
        loss_fn = CrossEntropyLoss()

        logits = jnp.array([[2.0, -2.0], [-2.0, 2.0]])
        labels = jnp.array([0, 1])

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        assert loss > 0

    def test_multiclass_classification(self) -> None:
        """Test cross-entropy for multiclass classification."""
        loss_fn = CrossEntropyLoss()

        logits = jnp.array([[2.0, 1.0, 0.1, -1.0, -2.0], [0.1, -1.0, 2.0, 1.0, -2.0], [-2.0, -1.0, 0.1, 1.0, 2.0]])
        labels = jnp.array([0, 2, 4])

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        assert loss > 0

    def test_batch_size_one(self) -> None:
        """Test cross-entropy with batch size of 1."""
        loss_fn = CrossEntropyLoss()

        logits = jnp.array([[2.0, 1.0, 0.1]])
        labels = jnp.array([0])

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        assert loss > 0

    def test_with_uniform_logits(self) -> None:
        """Test cross-entropy with uniform logits (maximum entropy)."""
        loss_fn = CrossEntropyLoss()

        # Uniform logits
        logits = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        labels = jnp.array([0, 1])

        loss = loss_fn(logits, labels)

        # For 3 classes, maximum entropy is log(3) ≈ 1.099
        assert loss.shape == ()
        assert 1.0 < loss < 1.2

    def test_reduction_consistency(self) -> None:
        """Test that sum reduction equals mean * batch_size."""
        logits = jnp.array([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0], [1.0, 0.1, 2.0]])
        labels = jnp.array([0, 1, 2])
        batch_size = len(labels)

        loss_mean = CrossEntropyLoss(reduce="mean")(logits, labels)
        loss_sum = CrossEntropyLoss(reduce="sum")(logits, labels)

        assert jnp.allclose(loss_sum, loss_mean * batch_size, rtol=1e-5)

    def test_label_types(self) -> None:
        """Test cross-entropy with different label types."""
        loss_fn = CrossEntropyLoss()

        logits = jnp.array([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]])

        # Test with regular int array
        labels_int = jnp.array([0, 1], dtype=jnp.int32)
        loss_int = loss_fn(logits, labels_int)

        # Test with long array
        labels_long = jnp.array([0, 1], dtype=jnp.int64)
        loss_long = loss_fn(logits, labels_long)

        assert loss_int.shape == ()
        assert loss_long.shape == ()
        assert jnp.allclose(loss_int, loss_long)
