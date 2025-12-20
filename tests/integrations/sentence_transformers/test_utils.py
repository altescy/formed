"""Tests for sentence_transformers utility functions."""

from sentence_transformers import SentenceTransformer

from formed.integrations.sentence_transformers.utils import load_sentence_transformer


class TestLoadSentenceTransformer:
    """Test the load_sentence_transformer function."""

    @staticmethod
    def test_load_model() -> None:
        """Test loading a sentence transformer model."""
        model = load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
        assert model is not None
        assert isinstance(model, SentenceTransformer)
        assert hasattr(model, "encode")

    @staticmethod
    def test_model_encode_functionality() -> None:
        """Test that the loaded model can encode sentences."""
        model = load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
        sentences = ["This is a test sentence", "Another test sentence"]
        embeddings = model.encode(sentences)
        assert embeddings is not None
        assert len(embeddings) == 2
        # Check that embeddings have the expected shape
        assert len(embeddings.shape) == 2

    @staticmethod
    def test_caching_returns_same_model() -> None:
        """Test that the LRU cache returns the same model for repeated calls."""
        model1 = load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
        model2 = load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
        # Should be the exact same object due to caching
        assert model1 is model2

    @staticmethod
    def test_caching_with_different_models_returns_different_objects() -> None:
        """Test that different model names result in different cached objects."""
        model1 = load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
        model2 = load_sentence_transformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
        # Different models should be different objects
        assert model1 is not model2
