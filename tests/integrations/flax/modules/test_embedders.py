import dataclasses

import numpy
import pytest

from formed.integrations.flax import use_rngs
from formed.integrations.flax.modules.embedders import AnalyzedTextEmbedder, EmbedderOutput, TokenEmbedder
from formed.integrations.flax.modules.vectorizers import BagOfEmbeddingsSequenceVectorizer
from formed.integrations.ml.types import IDSequenceBatch


@dataclasses.dataclass
class AnalyzedTextBatch:
    """Simple implementation of IAnalyzedTextBatch for testing."""

    surfaces: IDSequenceBatch
    postags: IDSequenceBatch
    characters: IDSequenceBatch

    def __len__(self) -> int:
        return len(self.surfaces)


class TestTokenEmbedder:
    def test_basic_embedding(self) -> None:
        """Test basic token embedding."""
        with use_rngs(0):
            embedder = TokenEmbedder(vocab_size=100, embedding_dim=64)

            token_ids = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            mask = numpy.array([[True, True, True, True], [True, True, True, False]])

            batch = IDSequenceBatch(ids=token_ids, mask=mask)
            output = embedder(batch)

            assert isinstance(output, EmbedderOutput)
            assert output.embeddings.shape == (2, 4, 64)
            assert output.mask.shape == (2, 4)
            assert numpy.array_equal(output.mask, mask)

    def test_output_dim(self) -> None:
        """Test that output dimension matches embedding dimension."""
        with use_rngs(0):
            embedder = TokenEmbedder(vocab_size=100, embedding_dim=128)
            assert embedder.get_output_dim() == 128

    def test_different_vocab_sizes(self) -> None:
        """Test with different vocabulary sizes."""
        with use_rngs(0):
            for vocab_size in [50, 100, 1000]:
                embedder = TokenEmbedder(vocab_size=vocab_size, embedding_dim=64)

                token_ids = numpy.array([[1, 2], [3, 4]])
                mask = numpy.array([[True, True], [True, True]])
                batch = IDSequenceBatch(ids=token_ids, mask=mask)

                output = embedder(batch)
                assert output.embeddings.shape == (2, 2, 64)

    def test_nested_token_ids_2d(self) -> None:
        """Test with 3D token IDs (e.g., character-level within words)."""
        with use_rngs(0):
            embedder = TokenEmbedder(vocab_size=256, embedding_dim=32)

            # (batch=2, seq_len=3, char_len=5)
            token_ids = numpy.ones((2, 3, 5), dtype=numpy.int32)
            mask = numpy.ones((2, 3, 5), dtype=numpy.bool_)

            batch = IDSequenceBatch(ids=token_ids, mask=mask)
            output = embedder(batch)

            # Should average over character dimension by default
            assert output.embeddings.shape == (2, 3, 32)
            assert output.mask.shape == (2, 3)

    def test_nested_with_vectorizer(self) -> None:
        """Test 3D token IDs with custom vectorizer."""
        with use_rngs(0):
            vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="max")
            embedder = TokenEmbedder(vocab_size=256, embedding_dim=32, vectorizer=vectorizer)

            # (batch=2, seq_len=3, char_len=5)
            token_ids = numpy.ones((2, 3, 5), dtype=numpy.int32)
            mask = numpy.ones((2, 3, 5), dtype=numpy.bool_)

            batch = IDSequenceBatch(ids=token_ids, mask=mask)
            output = embedder(batch)

            assert output.embeddings.shape == (2, 3, 32)
            assert output.mask.shape == (2, 3)

    def test_shape_mismatch_error(self) -> None:
        """Test that shape mismatch between ids and mask raises error."""
        with use_rngs(0):
            embedder = TokenEmbedder(vocab_size=100, embedding_dim=64)

            token_ids = numpy.array([[1, 2, 3]])
            mask = numpy.array([[True, True]])  # Different length

            batch = IDSequenceBatch(ids=token_ids, mask=mask)

            with pytest.raises(ValueError, match="must have the same shape"):
                embedder(batch)

    def test_invalid_ndim_error(self) -> None:
        """Test that 4D or higher token IDs raise error."""
        with use_rngs(0):
            embedder = TokenEmbedder(vocab_size=100, embedding_dim=64)

            # 4D tensor
            token_ids = numpy.ones((2, 3, 4, 5), dtype=numpy.int32)
            mask = numpy.ones((2, 3, 4, 5), dtype=numpy.bool_)

            batch = IDSequenceBatch(ids=token_ids, mask=mask)

            with pytest.raises(ValueError, match="must be of shape"):
                embedder(batch)


class TestAnalyzedTextEmbedder:
    def test_surface_only(self) -> None:
        """Test with only surface form embeddings."""
        with use_rngs(0):
            surface_embedder = TokenEmbedder(vocab_size=100, embedding_dim=64)
            embedder = AnalyzedTextEmbedder(surface=surface_embedder)

            surfaces = IDSequenceBatch(ids=numpy.array([[1, 2, 3]]), mask=numpy.array([[True, True, True]]))
            postags = IDSequenceBatch(ids=numpy.array([[0, 0, 0]]), mask=numpy.array([[True, True, True]]))
            characters = IDSequenceBatch(ids=numpy.array([[[0]]]), mask=numpy.array([[[True]]]))

            batch = AnalyzedTextBatch(surfaces=surfaces, postags=postags, characters=characters)
            output = embedder(batch)

            assert output.embeddings.shape == (1, 3, 64)
            assert embedder.get_output_dim() == 64

    def test_surface_and_postag(self) -> None:
        """Test with surface and POS tag embeddings."""
        with use_rngs(0):
            surface_embedder = TokenEmbedder(vocab_size=100, embedding_dim=64)
            postag_embedder = TokenEmbedder(vocab_size=50, embedding_dim=32)
            embedder = AnalyzedTextEmbedder(surface=surface_embedder, postag=postag_embedder)

            surfaces = IDSequenceBatch(ids=numpy.array([[1, 2, 3]]), mask=numpy.array([[True, True, True]]))
            postags = IDSequenceBatch(ids=numpy.array([[10, 11, 12]]), mask=numpy.array([[True, True, True]]))
            characters = IDSequenceBatch(ids=numpy.array([[[0]]]), mask=numpy.array([[[True]]]))

            batch = AnalyzedTextBatch(surfaces=surfaces, postags=postags, characters=characters)
            output = embedder(batch)

            # Should concatenate: 64 + 32 = 96
            assert output.embeddings.shape == (1, 3, 96)
            assert embedder.get_output_dim() == 96

    def test_all_embeddings(self) -> None:
        """Test with all embedding types (surface, postag, character)."""
        with use_rngs(0):
            surface_embedder = TokenEmbedder(vocab_size=100, embedding_dim=64)
            postag_embedder = TokenEmbedder(vocab_size=50, embedding_dim=32)
            character_embedder = TokenEmbedder(vocab_size=256, embedding_dim=16)
            embedder = AnalyzedTextEmbedder(
                surface=surface_embedder, postag=postag_embedder, character=character_embedder
            )

            surfaces = IDSequenceBatch(ids=numpy.array([[1, 2]]), mask=numpy.array([[True, True]]))
            postags = IDSequenceBatch(ids=numpy.array([[10, 11]]), mask=numpy.array([[True, True]]))
            characters = IDSequenceBatch(
                ids=numpy.array([[[1, 2, 3], [4, 5, 6]]]), mask=numpy.array([[[True, True, True], [True, True, True]]])
            )

            batch = AnalyzedTextBatch(surfaces=surfaces, postags=postags, characters=characters)
            output = embedder(batch)

            # Should concatenate: 64 + 32 + 16 = 112
            assert output.embeddings.shape == (1, 2, 112)
            assert embedder.get_output_dim() == 112

    def test_postag_only(self) -> None:
        """Test with only POS tag embeddings."""
        with use_rngs(0):
            postag_embedder = TokenEmbedder(vocab_size=50, embedding_dim=32)
            embedder = AnalyzedTextEmbedder(postag=postag_embedder)

            surfaces = IDSequenceBatch(ids=numpy.array([[0, 0]]), mask=numpy.array([[True, True]]))
            postags = IDSequenceBatch(ids=numpy.array([[10, 11]]), mask=numpy.array([[True, True]]))
            characters = IDSequenceBatch(ids=numpy.array([[[0]]]), mask=numpy.array([[[True]]]))

            batch = AnalyzedTextBatch(surfaces=surfaces, postags=postags, characters=characters)
            output = embedder(batch)

            assert output.embeddings.shape == (1, 2, 32)
            assert embedder.get_output_dim() == 32

    def test_character_only(self) -> None:
        """Test with only character embeddings."""
        with use_rngs(0):
            character_embedder = TokenEmbedder(vocab_size=256, embedding_dim=16)
            embedder = AnalyzedTextEmbedder(character=character_embedder)

            surfaces = IDSequenceBatch(ids=numpy.array([[0]]), mask=numpy.array([[True]]))
            postags = IDSequenceBatch(ids=numpy.array([[0]]), mask=numpy.array([[True]]))
            characters = IDSequenceBatch(ids=numpy.array([[[1, 2, 3]]]), mask=numpy.array([[[True, True, True]]]))

            batch = AnalyzedTextBatch(surfaces=surfaces, postags=postags, characters=characters)
            output = embedder(batch)

            assert output.embeddings.shape == (1, 1, 16)
            assert embedder.get_output_dim() == 16

    def test_no_embedders_error(self) -> None:
        """Test that creating embedder with no sub-embedders raises error."""
        with pytest.raises(ValueError, match="At least one embedder must be provided"):
            AnalyzedTextEmbedder()

    def test_mask_from_last_embedder(self) -> None:
        """Test that mask comes from the last processed embedder."""
        with use_rngs(0):
            surface_embedder = TokenEmbedder(vocab_size=100, embedding_dim=64)
            embedder = AnalyzedTextEmbedder(surface=surface_embedder)

            surfaces = IDSequenceBatch(ids=numpy.array([[1, 2, 3, 4]]), mask=numpy.array([[True, True, True, False]]))
            postags = IDSequenceBatch(ids=numpy.array([[0, 0, 0, 0]]), mask=numpy.array([[True, True, True, True]]))
            characters = IDSequenceBatch(ids=numpy.array([[[0]]]), mask=numpy.array([[[True]]]))

            batch = AnalyzedTextBatch(surfaces=surfaces, postags=postags, characters=characters)
            output = embedder(batch)

            # Mask should come from surface embedder
            assert numpy.array_equal(output.mask, surfaces.mask)
