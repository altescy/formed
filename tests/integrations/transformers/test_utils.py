"""Tests for transformers utility functions."""

from transformers import AutoModelForSequenceClassification
from formed.integrations.transformers.utils import (
    load_pretrained_tokenizer,
    load_pretrained_transformer,
)


class TestLoadPretrainedTransformer:
    """Test the load_pretrained_transformer function."""

    @staticmethod
    def test_load_model_with_default_auto_class() -> None:
        """Test loading a model with default AutoModel."""
        model = load_pretrained_transformer("hf-internal-testing/tiny-random-bert")
        assert model is not None
        assert hasattr(model, "forward")

    @staticmethod
    def test_load_model_with_custom_auto_class() -> None:
        """Test loading a model with a custom auto class."""
        model = load_pretrained_transformer(
            "hf-internal-testing/tiny-random-bert",
            auto_class=AutoModelForSequenceClassification,
        )
        assert model is not None
        assert hasattr(model, "forward")
        # Verify it's a sequence classification model
        assert hasattr(model, "num_labels")

    @staticmethod
    def test_load_model_with_submodule() -> None:
        """Test loading a specific submodule of a model."""
        model = load_pretrained_transformer(
            "hf-internal-testing/tiny-random-bert",
            submodule="embeddings",
        )
        assert model is not None
        # The submodule should be the embeddings module
        assert hasattr(model, "word_embeddings")

    @staticmethod
    def test_caching_returns_same_object() -> None:
        """Test that the LRU cache returns the same object for repeated calls."""
        model1 = load_pretrained_transformer("hf-internal-testing/tiny-random-bert")
        model2 = load_pretrained_transformer("hf-internal-testing/tiny-random-bert")
        # Should be the exact same object due to caching
        assert model1 is model2

    @staticmethod
    def test_caching_with_different_params_returns_different_objects() -> None:
        """Test that different parameters result in different cached objects."""
        model1 = load_pretrained_transformer("hf-internal-testing/tiny-random-bert")
        model2 = load_pretrained_transformer(
            "hf-internal-testing/tiny-random-bert",
            auto_class=AutoModelForSequenceClassification,
        )
        # Different auto_class should result in different objects
        assert model1 is not model2


class TestLoadPretrainedTokenizer:
    """Test the load_pretrained_tokenizer function."""

    @staticmethod
    def test_load_tokenizer() -> None:
        """Test loading a tokenizer."""
        tokenizer = load_pretrained_tokenizer("hf-internal-testing/tiny-random-bert")
        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")

    @staticmethod
    def test_tokenizer_functionality() -> None:
        """Test that the loaded tokenizer can encode and decode text."""
        tokenizer = load_pretrained_tokenizer("hf-internal-testing/tiny-random-bert")
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list)
        assert len(encoded) > 0

        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)

    @staticmethod
    def test_caching_returns_same_tokenizer() -> None:
        """Test that the LRU cache returns the same tokenizer for repeated calls."""
        tokenizer1 = load_pretrained_tokenizer("hf-internal-testing/tiny-random-bert")
        tokenizer2 = load_pretrained_tokenizer("hf-internal-testing/tiny-random-bert")
        # Should be the exact same object due to caching
        assert tokenizer1 is tokenizer2

    @staticmethod
    def test_caching_with_different_models_returns_different_tokenizers() -> None:
        """Test that different model names result in different cached tokenizers."""
        tokenizer1 = load_pretrained_tokenizer("hf-internal-testing/tiny-random-bert")
        tokenizer2 = load_pretrained_tokenizer("hf-internal-testing/tiny-random-gpt2")
        # Different models should have different tokenizers
        assert tokenizer1 is not tokenizer2
