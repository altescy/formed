"""Tests for transformers analyzers."""

from transformers import AutoTokenizer

from formed.integrations.ml.types import AnalyzedText
from formed.integrations.transformers.analyzers import PretrainedTransformerAnalyzer


class TestPretrainedTransformerAnalyzer:
    """Test the PretrainedTransformerAnalyzer class."""

    @staticmethod
    def test_initialization_from_string() -> None:
        """Test PretrainedTransformerAnalyzer initialization from tokenizer name string."""
        analyzer = PretrainedTransformerAnalyzer("hf-internal-testing/tiny-random-bert")

        # Verify tokenizer is loaded
        assert analyzer._tokenizer is not None
        assert hasattr(analyzer._tokenizer, "tokenize")

    @staticmethod
    def test_initialization_from_tokenizer() -> None:
        """Test PretrainedTransformerAnalyzer initialization from tokenizer instance."""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        analyzer = PretrainedTransformerAnalyzer(tokenizer)

        # Verify tokenizer is stored
        assert analyzer._tokenizer is tokenizer

    @staticmethod
    def test_call_returns_analyzed_text() -> None:
        """Test that calling analyzer returns AnalyzedText."""
        analyzer = PretrainedTransformerAnalyzer("hf-internal-testing/tiny-random-bert")
        result = analyzer("Hello, world!")

        # Verify return type
        assert isinstance(result, AnalyzedText)
        assert hasattr(result, "surfaces")
        assert hasattr(result, "postags")

    @staticmethod
    def test_surfaces_are_tokenized() -> None:
        """Test that surfaces contain tokenized text."""
        analyzer = PretrainedTransformerAnalyzer("hf-internal-testing/tiny-random-bert")
        result = analyzer("Hello world")

        # Verify surfaces is a sequence of strings
        assert isinstance(result.surfaces, (list, tuple))
        assert len(result.surfaces) > 0
        assert all(isinstance(token, str) for token in result.surfaces)

    @staticmethod
    def test_postags_is_none() -> None:
        """Test that postags field is None."""
        analyzer = PretrainedTransformerAnalyzer("hf-internal-testing/tiny-random-bert")
        result = analyzer("Hello world")

        # PretrainedTransformerAnalyzer only provides surfaces, not POS tags
        assert result.postags is None

    @staticmethod
    def test_empty_string() -> None:
        """Test analysis of empty string."""
        analyzer = PretrainedTransformerAnalyzer("hf-internal-testing/tiny-random-bert")
        result = analyzer("")

        # Empty string should return empty or minimal tokens
        assert isinstance(result.surfaces, (list, tuple))
        # Some tokenizers may add special tokens even for empty strings
        assert all(isinstance(token, str) for token in result.surfaces)

    @staticmethod
    def test_tokenization_consistency() -> None:
        """Test that same input produces same output."""
        analyzer = PretrainedTransformerAnalyzer("hf-internal-testing/tiny-random-bert")
        text = "Machine learning is amazing!"

        result1 = analyzer(text)
        result2 = analyzer(text)

        # Should produce identical tokenization
        assert result1.surfaces == result2.surfaces

    @staticmethod
    def test_different_texts_different_tokens() -> None:
        """Test that different texts produce different tokens."""
        analyzer = PretrainedTransformerAnalyzer("hf-internal-testing/tiny-random-bert")

        result1 = analyzer("Hello world")
        result2 = analyzer("Goodbye world")

        # Different inputs should generally produce different tokenization
        # (unless they happen to be identical after tokenization, which is unlikely)
        assert result1.surfaces != result2.surfaces

    @staticmethod
    def test_special_characters() -> None:
        """Test tokenization with special characters."""
        analyzer = PretrainedTransformerAnalyzer("hf-internal-testing/tiny-random-bert")
        result = analyzer("Hello, world! How are you?")

        # Should handle punctuation
        assert isinstance(result.surfaces, (list, tuple))
        assert len(result.surfaces) > 0
        assert all(isinstance(token, str) for token in result.surfaces)

    @staticmethod
    def test_unicode_text() -> None:
        """Test tokenization with unicode characters."""
        analyzer = PretrainedTransformerAnalyzer("hf-internal-testing/tiny-random-bert")
        result = analyzer("こんにちは世界")  # Japanese text

        # Should handle unicode
        assert isinstance(result.surfaces, (list, tuple))
        assert all(isinstance(token, str) for token in result.surfaces)

    @staticmethod
    def test_long_text() -> None:
        """Test tokenization with longer text."""
        analyzer = PretrainedTransformerAnalyzer("hf-internal-testing/tiny-random-bert")
        long_text = "This is a longer sentence with multiple words that should be tokenized properly. " * 5
        result = analyzer(long_text)

        # Should handle long text
        assert isinstance(result.surfaces, (list, tuple))
        assert len(result.surfaces) > 10  # Should have many tokens
        assert all(isinstance(token, str) for token in result.surfaces)

    @staticmethod
    def test_multiline_text() -> None:
        """Test tokenization with multiline text."""
        analyzer = PretrainedTransformerAnalyzer("hf-internal-testing/tiny-random-bert")
        text = "First line\nSecond line\nThird line"
        result = analyzer(text)

        # Should handle multiline text
        assert isinstance(result.surfaces, (list, tuple))
        assert all(isinstance(token, str) for token in result.surfaces)
