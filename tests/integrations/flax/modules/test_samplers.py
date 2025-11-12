import jax.numpy as jnp
from flax import nnx

from formed.integrations.flax.modules.samplers import (
    ArgmaxLabelSampler,
    MultinomialLabelSampler,
    MultinomialLabelSamplerParams,
)


class TestArgmaxLabelSampler:
    def test_basic_argmax(self) -> None:
        """Test basic argmax sampling."""
        sampler = ArgmaxLabelSampler()

        logits = jnp.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])

        labels = sampler(logits)

        assert labels.shape == (2,)
        assert labels[0] == 2  # Max is at index 2
        assert labels[1] == 0  # Max is at index 0

    def test_binary_classification(self) -> None:
        """Test argmax for binary classification."""
        sampler = ArgmaxLabelSampler()

        logits = jnp.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])

        labels = sampler(logits)

        assert labels.shape == (3,)
        assert labels[0] == 1
        assert labels[1] == 0
        assert labels[2] == 1

    def test_multiclass_classification(self) -> None:
        """Test argmax for multiclass classification."""
        sampler = ArgmaxLabelSampler()

        logits = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.3, 0.3, 0.4]])

        labels = sampler(logits)

        assert labels.shape == (3,)
        assert labels[0] == 4  # Max is at last position
        assert labels[1] == 0  # Max is at first position
        assert labels[2] == 4  # Max is at last position

    def test_negative_logits(self) -> None:
        """Test argmax with negative logits."""
        sampler = ArgmaxLabelSampler()

        logits = jnp.array([[-1.0, -2.0, -0.5], [-3.0, -1.0, -2.0]])

        labels = sampler(logits)

        assert labels.shape == (2,)
        assert labels[0] == 2  # -0.5 is the max
        assert labels[1] == 1  # -1.0 is the max

    def test_large_logits(self) -> None:
        """Test argmax with large logit values."""
        sampler = ArgmaxLabelSampler()

        logits = jnp.array([[100.0, 200.0, 150.0], [1000.0, 500.0, 750.0]])

        labels = sampler(logits)

        assert labels.shape == (2,)
        assert labels[0] == 1  # 200.0 is the max
        assert labels[1] == 0  # 1000.0 is the max

    def test_single_sample(self) -> None:
        """Test argmax with single sample."""
        sampler = ArgmaxLabelSampler()

        logits = jnp.array([[0.5, 1.5, 2.5]])

        labels = sampler(logits)

        assert labels.shape == (1,)
        assert labels[0] == 2

    def test_params_ignored(self) -> None:
        """Test that params argument is ignored (for interface compatibility)."""
        sampler = ArgmaxLabelSampler()

        logits = jnp.array([[1.0, 2.0, 3.0]])

        labels_with_params = sampler(logits, params=None)
        labels_without_params = sampler(logits)

        assert jnp.array_equal(labels_with_params, labels_without_params)


class TestMultinomialLabelSampler:
    def test_basic_sampling(self) -> None:
        """Test basic multinomial sampling."""
        sampler = MultinomialLabelSampler()

        logits = jnp.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        rngs = nnx.Rngs(0)

        params = MultinomialLabelSamplerParams(rngs=rngs)
        labels = sampler(logits, params)

        assert labels.shape == (2,)
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < 3)

    def test_deterministic_with_seed(self) -> None:
        """Test that same seed produces same results."""
        sampler = MultinomialLabelSampler()

        logits = jnp.array([[1.0, 2.0, 3.0]])

        params1 = MultinomialLabelSamplerParams(rngs=nnx.Rngs(42))
        labels1 = sampler(logits, params1)

        params2 = MultinomialLabelSamplerParams(rngs=nnx.Rngs(42))
        labels2 = sampler(logits, params2)

        assert jnp.array_equal(labels1, labels2)

    def test_different_seeds_different_results(self) -> None:
        """Test that different seeds can produce different results."""
        sampler = MultinomialLabelSampler()

        # Use moderate logits to allow for variation
        logits = jnp.array([[0.5, 0.6, 0.7]] * 100)  # Multiple samples for statistical variation

        params1 = MultinomialLabelSamplerParams(rngs=nnx.Rngs(0))
        labels1 = sampler(logits, params1)

        params2 = MultinomialLabelSamplerParams(rngs=nnx.Rngs(999))
        labels2 = sampler(logits, params2)

        # With enough samples, different seeds should produce different results
        assert not jnp.array_equal(labels1, labels2)

    def test_temperature_high(self) -> None:
        """Test sampling with high temperature (more random)."""
        sampler = MultinomialLabelSampler()

        logits = jnp.array([[1.0, 2.0, 3.0]] * 100)
        rngs = nnx.Rngs(0)

        params = MultinomialLabelSamplerParams(rngs=rngs, templerature=2.0)
        labels = sampler(logits, params)

        assert labels.shape == (100,)
        # With high temperature, should see more diversity in predictions
        unique_labels = jnp.unique(labels)
        assert len(unique_labels) > 1

    def test_temperature_low(self) -> None:
        """Test sampling with low temperature (more deterministic)."""
        sampler = MultinomialLabelSampler()

        logits = jnp.array([[1.0, 2.0, 3.0]] * 100)
        rngs = nnx.Rngs(0)

        params = MultinomialLabelSamplerParams(rngs=rngs, templerature=0.1)
        labels = sampler(logits, params)

        assert labels.shape == (100,)
        # With low temperature, should mostly pick the highest logit class
        # Most samples should be class 2 (highest logit)
        assert jnp.sum(labels == 2) > 50

    def test_temperature_one(self) -> None:
        """Test sampling with temperature=1.0 (default behavior)."""
        sampler = MultinomialLabelSampler()

        logits = jnp.array([[1.0, 2.0, 3.0]] * 100)
        rngs = nnx.Rngs(0)

        params = MultinomialLabelSamplerParams(rngs=rngs, templerature=1.0)
        labels = sampler(logits, params)

        assert labels.shape == (100,)
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < 3)

    def test_binary_classification(self) -> None:
        """Test multinomial sampling for binary classification."""
        sampler = MultinomialLabelSampler()

        logits = jnp.array([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]])
        rngs = nnx.Rngs(0)

        params = MultinomialLabelSamplerParams(rngs=rngs)
        labels = sampler(logits, params)

        assert labels.shape == (3,)
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < 2)

    def test_multiclass_classification(self) -> None:
        """Test multinomial sampling for multiclass classification."""
        sampler = MultinomialLabelSampler()

        logits = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])
        rngs = nnx.Rngs(0)

        params = MultinomialLabelSamplerParams(rngs=rngs)
        labels = sampler(logits, params)

        assert labels.shape == (2,)
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < 5)

    def test_uniform_logits(self) -> None:
        """Test multinomial sampling with uniform logits."""
        sampler = MultinomialLabelSampler()

        # Uniform logits should give roughly equal probability to all classes
        logits = jnp.array([[0.0, 0.0, 0.0]] * 300)
        rngs = nnx.Rngs(42)

        params = MultinomialLabelSamplerParams(rngs=rngs)
        labels = sampler(logits, params)

        # With uniform probabilities, each class should appear roughly equally
        counts = jnp.array([(labels == i).sum() for i in range(3)])
        # Each class should appear at least 50 times out of 300 (very loose bound)
        assert jnp.all(counts > 50)

    def test_negative_logits(self) -> None:
        """Test multinomial sampling with negative logits."""
        sampler = MultinomialLabelSampler()

        logits = jnp.array([[-1.0, -2.0, -0.5], [-3.0, -1.0, -2.0]])
        rngs = nnx.Rngs(0)

        params = MultinomialLabelSamplerParams(rngs=rngs)
        labels = sampler(logits, params)

        assert labels.shape == (2,)
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < 3)

    def test_default_params(self) -> None:
        """Test that sampler works with default params (None) when rngs are available."""
        from formed.integrations.flax import use_rngs

        sampler = MultinomialLabelSampler()

        logits = jnp.array([[1.0, 2.0, 3.0]])

        # Should use default params internally with rngs context
        with use_rngs(0):
            labels = sampler(logits, params=None)

            assert labels.shape == (1,)
            assert jnp.all(labels >= 0)
            assert jnp.all(labels < 3)

    def test_single_sample(self) -> None:
        """Test multinomial sampling with single sample."""
        sampler = MultinomialLabelSampler()

        logits = jnp.array([[0.5, 1.5, 2.5]])
        rngs = nnx.Rngs(0)

        params = MultinomialLabelSamplerParams(rngs=rngs)
        labels = sampler(logits, params)

        assert labels.shape == (1,)
        assert 0 <= labels[0] < 3

    def test_high_dimension_stability(self) -> None:
        """Test that sampling is stable with many classes."""
        sampler = MultinomialLabelSampler()

        num_classes = 100
        logits = jnp.ones((5, num_classes))
        rngs = nnx.Rngs(0)

        params = MultinomialLabelSamplerParams(rngs=rngs)
        labels = sampler(logits, params)

        assert labels.shape == (5,)
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < num_classes)
