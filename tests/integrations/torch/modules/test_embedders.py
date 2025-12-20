"""Tests for embedders module."""

from dataclasses import dataclass

import pytest
import torch

from formed.integrations.torch.initializers import UniformTensorInitializer
from formed.integrations.torch.modules.embedders import (
    AnalyzedTextEmbedder,
    EmbedderOutput,
    PassThroughEmbedder,
    PretrainedTransformerEmbedder,
    TokenEmbedder,
)


@dataclass
class IDSequenceBatch:
    """Simple implementation of IIDSequenceBatch for testing."""

    ids: torch.Tensor
    mask: torch.Tensor

    def __len__(self) -> int:
        return self.ids.shape[0]


@dataclass
class VariableTensorBatch:
    tensor: torch.Tensor
    mask: torch.Tensor

    def __len__(self) -> int:
        return self.tensor.shape[0]


@dataclass
class AnalyzedTextBatch:
    """Simple implementation of IAnalyzedTextBatch for testing."""

    surfaces: IDSequenceBatch | None = None
    postags: IDSequenceBatch | None = None
    characters: IDSequenceBatch | None = None
    token_vectors: VariableTensorBatch | None = None

    def __len__(self) -> int:
        if self.surfaces is not None:
            return len(self.surfaces)
        if self.postags is not None:
            return len(self.postags)
        if self.characters is not None:
            return len(self.characters)
        if self.token_vectors is not None:
            return len(self.token_vectors)
        return 0


class TestTokenEmbedder:
    def test_initialization(self):
        """Test TokenEmbedder initialization."""
        vocab_size = 1000
        embedding_dim = 128
        initializer = UniformTensorInitializer(shape=(vocab_size, embedding_dim))

        embedder = TokenEmbedder(initializer=initializer, padding_idx=0)

        assert embedder.get_output_dim() == embedding_dim
        assert embedder._embedding.num_embeddings == vocab_size
        assert embedder._embedding.embedding_dim == embedding_dim

    def test_forward_2d(self):
        """Test forward pass with 2D inputs (batch_size, seq_len)."""
        vocab_size = 100
        embedding_dim = 64
        batch_size = 8
        seq_len = 16

        initializer = UniformTensorInitializer(shape=(vocab_size, embedding_dim))
        embedder = TokenEmbedder(initializer=initializer, padding_idx=0)

        # Create 2D token IDs and mask
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)

        batch = IDSequenceBatch(ids=token_ids, mask=mask)
        output = embedder(batch)

        assert isinstance(output, EmbedderOutput)
        assert output.embeddings.shape == (batch_size, seq_len, embedding_dim)
        assert output.mask.shape == (batch_size, seq_len)

    def test_forward_3d_with_average_pooling(self):
        """Test forward pass with 3D inputs using average pooling."""
        vocab_size = 100
        embedding_dim = 32
        batch_size = 4
        seq_len = 10
        char_len = 8

        initializer = UniformTensorInitializer(shape=(vocab_size, embedding_dim))
        embedder = TokenEmbedder(initializer=initializer, padding_idx=0)

        # Create 3D token IDs (character-level) and mask
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len, char_len))
        mask = torch.ones(batch_size, seq_len, char_len)
        # Mask out some characters
        mask[:, :, -2:] = 0

        batch = IDSequenceBatch(ids=token_ids, mask=mask)
        output = embedder(batch)

        # Should pool characters to word-level
        assert output.embeddings.shape == (batch_size, seq_len, embedding_dim)
        assert output.mask.shape == (batch_size, seq_len)

    def test_padding_mask(self):
        """Test that padding is handled correctly."""
        vocab_size = 50
        embedding_dim = 32
        batch_size = 4
        seq_len = 8

        initializer = UniformTensorInitializer(shape=(vocab_size, embedding_dim))
        embedder = TokenEmbedder(initializer=initializer, padding_idx=0)

        # Create token IDs with padding (0s)
        token_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        token_ids[:, seq_len // 2 :] = 0  # Pad second half

        # Create corresponding mask
        mask = (token_ids != 0).float()

        batch = IDSequenceBatch(ids=token_ids, mask=mask)
        output = embedder(batch)

        assert output.embeddings.shape == (batch_size, seq_len, embedding_dim)
        assert torch.equal(output.mask, mask.bool())

    def test_freeze_embeddings(self):
        """Test that embeddings can be frozen."""
        vocab_size = 50
        embedding_dim = 32

        initializer = UniformTensorInitializer(shape=(vocab_size, embedding_dim))
        embedder = TokenEmbedder(initializer=initializer, padding_idx=0, freeze=True)

        # Check that embeddings are frozen
        assert not embedder._embedding.weight.requires_grad

    def test_invalid_shape_raises_error(self):
        """Test that invalid input shapes raise errors."""
        vocab_size = 50
        embedding_dim = 32

        initializer = UniformTensorInitializer(shape=(vocab_size, embedding_dim))
        embedder = TokenEmbedder(initializer=initializer, padding_idx=0)

        # 4D input should raise error
        token_ids = torch.randint(0, vocab_size, (2, 4, 8, 10))
        mask = torch.ones(2, 4, 8, 10)
        batch = IDSequenceBatch(ids=token_ids, mask=mask)

        with pytest.raises(ValueError, match="must be of shape"):
            embedder(batch)

    def test_shape_mismatch_raises_error(self):
        """Test that shape mismatch between ids and mask raises error."""
        vocab_size = 50
        embedding_dim = 32

        initializer = UniformTensorInitializer(shape=(vocab_size, embedding_dim))
        embedder = TokenEmbedder(initializer=initializer, padding_idx=0)

        # Mismatched shapes
        token_ids = torch.randint(0, vocab_size, (4, 8))
        mask = torch.ones(4, 10)  # Different seq_len
        batch = IDSequenceBatch(ids=token_ids, mask=mask)

        with pytest.raises(ValueError, match="must have the same shape"):
            embedder(batch)


class TestAnalyzedTextEmbedder:
    def test_initialization_with_all_embedders(self):
        """Test AnalyzedTextEmbedder initialization with all embedders."""
        vocab_size = 100
        embedding_dim = 32

        surface_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(vocab_size, embedding_dim)))
        postag_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(50, 16)))
        char_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(256, 16)))

        embedder = AnalyzedTextEmbedder(
            surface=surface_embedder,
            postag=postag_embedder,
            character=char_embedder,
        )

        # Total output dim should be sum of all embeddings
        assert embedder.get_output_dim() == 32 + 16 + 16

    def test_initialization_with_single_embedder(self):
        """Test AnalyzedTextEmbedder with only surface embedder."""
        vocab_size = 100
        embedding_dim = 64

        surface_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(vocab_size, embedding_dim)))

        embedder = AnalyzedTextEmbedder(surface=surface_embedder)

        assert embedder.get_output_dim() == embedding_dim

    def test_initialization_without_embedders_raises_error(self):
        """Test that initialization without any embedder raises error."""
        with pytest.raises(ValueError, match="At least one embedder must be provided"):
            AnalyzedTextEmbedder()

    def test_forward_with_all_features(self):
        """Test forward pass with all linguistic features."""
        batch_size = 4
        seq_len = 10

        surface_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(1000, 64)))
        postag_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(50, 16)))
        char_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(256, 16)))

        embedder = AnalyzedTextEmbedder(
            surface=surface_embedder,
            postag=postag_embedder,
            character=char_embedder,
        )

        # Create input batch
        surfaces = IDSequenceBatch(
            ids=torch.randint(0, 1000, (batch_size, seq_len)),
            mask=torch.ones(batch_size, seq_len),
        )
        postags = IDSequenceBatch(
            ids=torch.randint(0, 50, (batch_size, seq_len)),
            mask=torch.ones(batch_size, seq_len),
        )
        characters = IDSequenceBatch(
            ids=torch.randint(0, 256, (batch_size, seq_len, 8)),
            mask=torch.ones(batch_size, seq_len, 8),
        )

        batch = AnalyzedTextBatch(surfaces=surfaces, postags=postags, characters=characters)
        output = embedder(batch)

        # Output should concatenate all embeddings
        assert output.embeddings.shape == (batch_size, seq_len, 64 + 16 + 16)
        assert output.mask.shape == (batch_size, seq_len)

    def test_forward_with_partial_features(self):
        """Test forward pass with only some features present."""
        batch_size = 4
        seq_len = 10

        surface_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(1000, 64)))
        postag_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(50, 16)))

        embedder = AnalyzedTextEmbedder(surface=surface_embedder, postag=postag_embedder)

        # Only provide surface and postag, not characters
        surfaces = IDSequenceBatch(
            ids=torch.randint(0, 1000, (batch_size, seq_len)),
            mask=torch.ones(batch_size, seq_len),
        )
        postags = IDSequenceBatch(
            ids=torch.randint(0, 50, (batch_size, seq_len)),
            mask=torch.ones(batch_size, seq_len),
        )

        batch = AnalyzedTextBatch(surfaces=surfaces, postags=postags)
        output = embedder(batch)

        assert output.embeddings.shape == (batch_size, seq_len, 64 + 16)
        assert output.mask.shape == (batch_size, seq_len)

    def test_forward_without_embeddings_raises_error(self):
        """Test that forward without any embeddings raises error."""
        surface_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(1000, 64)))

        embedder = AnalyzedTextEmbedder(surface=surface_embedder)

        # Create batch with no data
        batch = AnalyzedTextBatch()

        with pytest.raises(ValueError, match="No embeddings were computed"):
            embedder(batch)

    def test_concatenation_order(self):
        """Test that embeddings are concatenated in the correct order."""
        batch_size = 2
        seq_len = 4

        # Create embedders with different dimensions for easy verification
        surface_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(100, 10)))
        postag_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(50, 5)))

        embedder = AnalyzedTextEmbedder(surface=surface_embedder, postag=postag_embedder)

        surfaces = IDSequenceBatch(
            ids=torch.randint(0, 100, (batch_size, seq_len)),
            mask=torch.ones(batch_size, seq_len),
        )
        postags = IDSequenceBatch(
            ids=torch.randint(0, 50, (batch_size, seq_len)),
            mask=torch.ones(batch_size, seq_len),
        )

        batch = AnalyzedTextBatch(surfaces=surfaces, postags=postags)
        output = embedder(batch)

        # Verify total dimension
        assert output.embeddings.shape == (batch_size, seq_len, 15)

        # Verify surface embeddings are in first 10 dimensions
        surface_output = surface_embedder(surfaces)
        assert torch.equal(output.embeddings[:, :, :10], surface_output.embeddings)

        # Verify postag embeddings are in last 5 dimensions
        postag_output = postag_embedder(postags)
        assert torch.equal(output.embeddings[:, :, 10:], postag_output.embeddings)

    def test_forward_with_token_vectors(self):
        """Test forward pass with token-level dense vectors."""
        batch_size = 2
        seq_len = 4
        vector_dim = 8

        surface_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(100, 16)))
        token_vector_embedder = PassThroughEmbedder()

        embedder = AnalyzedTextEmbedder(surface=surface_embedder, token_vector=token_vector_embedder)

        # Create input batch
        surfaces = IDSequenceBatch(
            ids=torch.randint(0, 100, (batch_size, seq_len)),
            mask=torch.ones(batch_size, seq_len),
        )
        token_vectors = VariableTensorBatch(
            tensor=torch.randn(batch_size, seq_len, vector_dim),
            mask=torch.ones(batch_size, seq_len),
        )

        batch = AnalyzedTextBatch(surfaces=surfaces, token_vectors=token_vectors)
        output = embedder(batch)

        # Output should concatenate surface embeddings and token vectors
        assert output.embeddings.shape == (batch_size, seq_len, 16 + vector_dim)
        assert output.mask.shape == (batch_size, seq_len)

    def test_forward_with_all_features_including_token_vectors(self):
        """Test forward pass with all features including token vectors."""
        batch_size = 2
        seq_len = 4
        vector_dim = 8

        surface_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(100, 16)))
        postag_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(20, 8)))
        char_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(50, 8)))
        token_vector_embedder = PassThroughEmbedder()

        embedder = AnalyzedTextEmbedder(
            surface=surface_embedder,
            postag=postag_embedder,
            character=char_embedder,
            token_vector=token_vector_embedder,
        )

        # Create input batch with all features
        surfaces = IDSequenceBatch(
            ids=torch.randint(0, 100, (batch_size, seq_len)),
            mask=torch.ones(batch_size, seq_len),
        )
        postags = IDSequenceBatch(
            ids=torch.randint(0, 20, (batch_size, seq_len)),
            mask=torch.ones(batch_size, seq_len),
        )
        characters = IDSequenceBatch(
            ids=torch.randint(0, 50, (batch_size, seq_len, 4)),
            mask=torch.ones(batch_size, seq_len, 4),
        )
        token_vectors = VariableTensorBatch(
            tensor=torch.randn(batch_size, seq_len, vector_dim),
            mask=torch.ones(batch_size, seq_len),
        )

        batch = AnalyzedTextBatch(
            surfaces=surfaces, postags=postags, characters=characters, token_vectors=token_vectors
        )
        output = embedder(batch)

        # Output should concatenate all embeddings: 16 + 8 + 8 + 8 = 40
        assert output.embeddings.shape == (batch_size, seq_len, 16 + 8 + 8 + vector_dim)
        assert output.mask.shape == (batch_size, seq_len)

    def test_forward_with_only_token_vectors_raises_error(self):
        """Test that token vectors alone without surface/postag/char raises error."""
        token_vector_embedder = PassThroughEmbedder()

        # This should raise ValueError since at least one of surface/postag/character is required
        with pytest.raises(ValueError, match="At least one embedder must be provided"):
            AnalyzedTextEmbedder(token_vector=token_vector_embedder)

    def test_forward_without_token_vectors_when_configured(self):
        """Test forward pass when token_vector embedder is configured but not provided in batch."""
        batch_size = 2
        seq_len = 4

        surface_embedder = TokenEmbedder(initializer=UniformTensorInitializer(shape=(100, 16)))
        token_vector_embedder = PassThroughEmbedder()

        embedder = AnalyzedTextEmbedder(surface=surface_embedder, token_vector=token_vector_embedder)

        # Create batch WITHOUT token_vectors
        surfaces = IDSequenceBatch(
            ids=torch.randint(0, 100, (batch_size, seq_len)),
            mask=torch.ones(batch_size, seq_len),
        )

        batch = AnalyzedTextBatch(surfaces=surfaces)
        output = embedder(batch)

        # Output should only have surface embeddings (no token vectors added)
        assert output.embeddings.shape == (batch_size, seq_len, 16)
        assert output.mask.shape == (batch_size, seq_len)


class TestPretrainedTransformerEmbedder:
    @pytest.fixture(scope="class")
    def tiny_model_name(self) -> str:
        """Fixture providing a tiny pretrained model for testing."""
        return "hf-internal-testing/tiny-random-bert"

    def test_initialization_from_string(self, tiny_model_name: str):
        """Test PretrainedTransformerEmbedder initialization from model name string."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name)

        # Verify model is loaded and has config
        assert embedder._model is not None
        assert hasattr(embedder._model, "config")
        assert embedder.get_output_dim() > 0

    def test_initialization_with_auto_class_string(self, tiny_model_name: str):
        """Test initialization with auto_class as string."""
        embedder = PretrainedTransformerEmbedder(
            model=tiny_model_name,
            auto_class="transformers:AutoModel",
        )

        assert embedder._model is not None
        assert embedder.get_output_dim() > 0

    def test_initialization_with_freeze(self, tiny_model_name: str):
        """Test that model parameters can be frozen."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name, freeze=True)

        # Check that all parameters are frozen
        for param in embedder._model.parameters():
            assert not param.requires_grad

    def test_initialization_without_freeze(self, tiny_model_name: str):
        """Test that model parameters are trainable by default."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name, freeze=False)

        # Check that parameters are trainable
        trainable_params = [p for p in embedder._model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0

    def test_forward(self, tiny_model_name: str):
        """Test forward pass with a pretrained transformer."""
        batch_size = 4
        seq_len = 16

        embedder = PretrainedTransformerEmbedder(model=tiny_model_name)

        # Create input batch - using vocab_size from the model's config
        vocab_size = embedder.get_vocab_size()
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)

        batch = IDSequenceBatch(ids=token_ids, mask=mask)
        output = embedder(batch)

        assert isinstance(output, EmbedderOutput)
        assert output.embeddings.shape == (batch_size, seq_len, embedder.get_output_dim())
        assert output.mask.shape == (batch_size, seq_len)

    def test_forward_with_padding(self, tiny_model_name: str):
        """Test forward pass with padding in the sequence."""
        batch_size = 4
        seq_len = 16

        embedder = PretrainedTransformerEmbedder(model=tiny_model_name)

        # Create input with padding
        vocab_size = embedder.get_vocab_size()
        token_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        token_ids[:, seq_len // 2 :] = 0  # Pad second half

        # Create mask (0 for padded positions)
        mask = (token_ids != 0).float()

        batch = IDSequenceBatch(ids=token_ids, mask=mask)
        output = embedder(batch)

        assert output.embeddings.shape == (batch_size, seq_len, embedder.get_output_dim())
        assert torch.equal(output.mask, mask)

    def test_gradient_flow(self, tiny_model_name: str):
        """Test that gradients flow correctly when not frozen."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name, freeze=False)
        embedder.train()  # Ensure model is in training mode

        batch_size = 2
        seq_len = 8
        vocab_size = embedder.get_vocab_size()

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)

        batch = IDSequenceBatch(ids=token_ids, mask=mask)
        output = embedder(batch)

        # Verify output requires grad
        assert output.embeddings.requires_grad

        # Compute a simple loss and backprop
        loss = output.embeddings.sum()
        loss.backward()

        # Check that some parameters have gradients
        params_with_grad = [p for p in embedder._model.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0

    def test_no_gradient_when_frozen(self, tiny_model_name: str):
        """Test that no gradients are computed when model is frozen."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name, freeze=True)

        batch_size = 2
        seq_len = 8
        vocab_size = embedder.get_vocab_size()

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)

        batch = IDSequenceBatch(ids=token_ids, mask=mask)
        output = embedder(batch)

        # Verify output does not require grad (frozen)
        assert not output.embeddings.requires_grad

        # Cannot backprop through frozen model
        # Just verify all parameters are frozen
        for param in embedder._model.parameters():
            assert not param.requires_grad

    def test_different_sequence_lengths(self, tiny_model_name: str):
        """Test that embedder handles different sequence lengths correctly."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name)
        vocab_size = embedder.get_vocab_size()

        for seq_len in [4, 8, 16, 32]:
            batch_size = 2
            token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            mask = torch.ones(batch_size, seq_len)

            batch = IDSequenceBatch(ids=token_ids, mask=mask)
            output = embedder(batch)

            assert output.embeddings.shape == (batch_size, seq_len, embedder.get_output_dim())
            assert output.mask.shape == (batch_size, seq_len)

    def test_eval_mode_stays_in_eval(self, tiny_model_name: str):
        """Test that eval_mode=True keeps model in eval mode even after .train() call."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name, eval_mode=True)

        # Model should start in eval mode
        assert not embedder._model.training

        # Call train() on embedder
        embedder.train()

        # Model should still be in eval mode due to eval_mode=True
        assert not embedder._model.training
        assert embedder.training  # Embedder itself is in training mode

    def test_eval_mode_false_allows_training(self, tiny_model_name: str):
        """Test that eval_mode=False allows normal training mode switching."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name, eval_mode=False)

        # Start in training mode
        embedder.train()
        assert embedder._model.training

        # Switch to eval mode
        embedder.eval()
        assert not embedder._model.training

        # Switch back to training mode
        embedder.train()
        assert embedder._model.training

    def test_layer_to_use_embeddings(self, tiny_model_name: str):
        """Test layer_to_use='embeddings' returns only embedding layer output."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name, layer_to_use="embeddings")

        batch_size = 2
        seq_len = 8
        vocab_size = embedder.get_vocab_size()

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)

        batch = IDSequenceBatch(ids=token_ids, mask=mask)
        output = embedder(batch)

        # Should still output correct shape
        assert output.embeddings.shape == (batch_size, seq_len, embedder.get_output_dim())
        assert output.mask.shape == (batch_size, seq_len)

        # Verify that _model is now wrapped as _Embedding
        assert isinstance(embedder._model, PretrainedTransformerEmbedder._Embedding)

    def test_layer_to_use_all_with_scalar_mix(self, tiny_model_name: str):
        """Test layer_to_use='all' uses ScalarMix to combine all hidden layers."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name, layer_to_use="all")

        # ScalarMix should be initialized
        assert embedder._scalar_mix is not None

        batch_size = 2
        seq_len = 8
        vocab_size = embedder.get_vocab_size()

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)

        batch = IDSequenceBatch(ids=token_ids, mask=mask)
        output = embedder(batch)

        # Should output correct shape
        assert output.embeddings.shape == (batch_size, seq_len, embedder.get_output_dim())
        assert output.mask.shape == (batch_size, seq_len)

        # Verify model config has output_hidden_states enabled
        # Note: _model is a PreTrainedModel, not _Embedding wrapper
        assert hasattr(embedder._model, "config")
        assert embedder._model.config.output_hidden_states  # type: ignore[attr-defined]

    def test_layer_to_use_last_default(self, tiny_model_name: str):
        """Test layer_to_use='last' (default) uses last hidden state."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name, layer_to_use="last")

        # ScalarMix should not be initialized
        assert embedder._scalar_mix is None

        batch_size = 2
        seq_len = 8
        vocab_size = embedder.get_vocab_size()

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)

        batch = IDSequenceBatch(ids=token_ids, mask=mask)
        output = embedder(batch)

        # Should output correct shape
        assert output.embeddings.shape == (batch_size, seq_len, embedder.get_output_dim())
        assert output.mask.shape == (batch_size, seq_len)

    def test_gradient_checkpointing_enabled(self, tiny_model_name: str):
        """Test that gradient_checkpointing=True enables gradient checkpointing."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name, gradient_checkpointing=True)

        # Verify config is updated
        assert hasattr(embedder._model, "config")
        assert embedder._model.config.gradient_checkpointing is True  # type: ignore[attr-defined]

    def test_gradient_checkpointing_disabled(self, tiny_model_name: str):
        """Test that gradient_checkpointing=False disables gradient checkpointing."""
        embedder = PretrainedTransformerEmbedder(model=tiny_model_name, gradient_checkpointing=False)

        # Verify config is updated
        assert hasattr(embedder._model, "config")
        assert embedder._model.config.gradient_checkpointing is False  # type: ignore[attr-defined]

    def test_gradient_checkpointing_none_preserves_default(self, tiny_model_name: str):
        """Test that gradient_checkpointing=None preserves model's default config."""
        # Create two embedders - one with None, one without the parameter
        embedder1 = PretrainedTransformerEmbedder(model=tiny_model_name, gradient_checkpointing=None)
        embedder2 = PretrainedTransformerEmbedder(model=tiny_model_name)

        # Both should have the same gradient_checkpointing setting (the default)
        # Note: the default value depends on the model, so we just check they match
        assert hasattr(embedder1._model, "config")
        assert hasattr(embedder2._model, "config")
        assert embedder1._model.config.gradient_checkpointing == embedder2._model.config.gradient_checkpointing  # type: ignore[attr-defined]
