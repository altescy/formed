"""Text embedding modules for PyTorch models.

This module provides embedders that convert tokenized text into dense vector
representations. Embedders handle various text representations including
surface forms, part-of-speech tags, and character sequences.

Key Components:
    - BaseEmbedder: Abstract base class for all embedders
    - TokenEmbedder: Embeds token ID sequences into dense vectors
    - AnalyzedTextEmbedder: Combines multiple embedding types (surface, POS, chars)

Features:
    - Support for nested token sequences (e.g., word -> character)
    - Automatic masking and padding handling
    - Configurable vectorization for character-level embeddings
    - Concatenation of multiple embedding types

Example:
    >>> from formed.integrations.torch.modules import TokenEmbedder, AnalyzedTextEmbedder
    >>> import torch.nn as nn
    >>>
    >>> # Simple token embedder
    >>> embedder = TokenEmbedder(
    ...     vocab_size=10000,
    ...     embedding_dim=128
    ... )
    >>>
    >>> # Multi-feature embedder
    >>> embedder = AnalyzedTextEmbedder(
    ...     surface=TokenEmbedder(vocab_size=10000, embedding_dim=128),
    ...     postag=TokenEmbedder(vocab_size=50, embedding_dim=32)
    ... )

"""

import abc
from typing import Any, Generic, NamedTuple, Protocol, runtime_checkable

import torch
import torch.nn as nn
from colt import Registrable
from typing_extensions import TypeVar

from ..types import TensorCompatibleT
from ..utils import ensure_torch_tensor
from .vectorizers import BaseSequenceVectorizer

_TextBatchT = TypeVar("_TextBatchT")


@runtime_checkable
class IIDSequenceBatch(Protocol[TensorCompatibleT]):
    """Protocol for token ID sequence batches.

    Attributes:
        ids: Token IDs of shape (batch_size, seq_len) or (batch_size, seq_len, char_len).
        mask: Attention mask of shape (batch_size, seq_len) or (batch_size, seq_len, char_len).

    """

    ids: TensorCompatibleT
    mask: TensorCompatibleT

    def __len__(self) -> int: ...


SurfaceBatchT = TypeVar("SurfaceBatchT", bound=IIDSequenceBatch, default=Any)
PostagBatchT = TypeVar("PostagBatchT", bound=IIDSequenceBatch | None, default=Any)
CharacterBatchT = TypeVar("CharacterBatchT", bound=IIDSequenceBatch | None, default=Any)


@runtime_checkable
class IAnalyzedTextBatch(Protocol[SurfaceBatchT, PostagBatchT, CharacterBatchT]):
    """Protocol for analyzed text batches with multiple linguistic features.

    Attributes:
        surfaces: Surface form token IDs.
        postags: Part-of-speech tag IDs (optional).
        characters: Character sequence IDs (optional).

    """

    surfaces: SurfaceBatchT
    postags: PostagBatchT
    characters: CharacterBatchT


class EmbedderOutput(NamedTuple):
    """Output from an embedder.

    Attributes:
        embeddings: Dense embeddings of shape (batch_size, seq_len, embedding_dim).
        mask: Attention mask of shape (batch_size, seq_len).

    """

    embeddings: torch.Tensor
    mask: torch.Tensor


class BaseEmbedder(nn.Module, Registrable, Generic[_TextBatchT], abc.ABC):
    """Abstract base class for text embedders.

    Embedders convert tokenized text into dense vector representations.
    They output both embeddings and attention masks.

    Type Parameters:
        _TextBatchT: Type of input batch (e.g., IIDSequenceBatch, IAnalyzedTextBatch).

    """

    @abc.abstractmethod
    def forward(self, inputs: _TextBatchT) -> EmbedderOutput:
        """Embed input tokens into dense vectors.

        Args:
            inputs: Batch of tokenized text.

        Returns:
            EmbedderOutput containing embeddings and mask.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_output_dim(self) -> int:
        """Get the output embedding dimension.

        Returns:
            Embedding dimension.

        """
        raise NotImplementedError

    def __call__(self, inputs: _TextBatchT) -> EmbedderOutput:
        return super().__call__(inputs)


@BaseEmbedder.register("token")
class TokenEmbedder(BaseEmbedder[IIDSequenceBatch]):
    """Embedder for token ID sequences.

    This embedder converts token IDs into dense embeddings using a learned
    embedding matrix. It supports both 2D (batch_size, seq_len) and 3D
    (batch_size, seq_len, char_len) token ID tensors.

    For 3D inputs (e.g., character-level tokens within words), the embedder
    can either average the embeddings or apply a custom vectorizer.

    Args:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of the embedding vectors.
        padding_idx: Index of the padding token (default: 0).
        vectorizer: Optional vectorizer for 3D inputs (character sequences).

    Example:
        >>> # Simple word embeddings
        >>> embedder = TokenEmbedder(vocab_size=10000, embedding_dim=128)
        >>> output = embedder(word_ids_batch)
        >>>
        >>> # Character-level embeddings with pooling
        >>> from formed.integrations.torch.modules import BagOfEmbeddingsSequenceVectorizer
        >>> embedder = TokenEmbedder(
        ...     vocab_size=256,
        ...     embedding_dim=32,
        ...     vectorizer=BagOfEmbeddingsSequenceVectorizer(pooling="max")
        ... )

    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        padding_idx: int = 0,
        vectorizer: BaseSequenceVectorizer | None = None,
    ) -> None:
        super().__init__()
        self._embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self._vectorizer = vectorizer

    def forward(self, inputs: IIDSequenceBatch) -> EmbedderOutput:
        token_ids = ensure_torch_tensor(inputs.ids)
        mask = ensure_torch_tensor(inputs.mask).bool()

        nested = False
        if token_ids.ndim > 2:
            if token_ids.ndim != 3:
                raise ValueError("Token ids must be of shape (batch_size, seq_len) or (batch_size, seq_len, char_len)")
            nested = True

        if token_ids.shape != mask.shape:
            raise ValueError(f"Token ids and mask must have the same shape, got {token_ids.shape} and {mask.shape}")

        embeddings = self._embedding(token_ids)

        if nested:
            if self._vectorizer is None:
                # Average pooling over character dimension
                embeddings = (embeddings * mask.unsqueeze(-1)).sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
                mask = mask.any(dim=-1)
            else:
                # Flatten batch and sequence dimensions for vectorizer
                batch_size, seq_len, char_len = token_ids.shape
                flat_embeddings = embeddings.view(batch_size * seq_len, char_len, -1)
                flat_mask = mask.view(batch_size * seq_len, char_len)

                # Apply vectorizer
                flat_embeddings = self._vectorizer(flat_embeddings, mask=flat_mask)

                # Reshape back
                embeddings = flat_embeddings.view(batch_size, seq_len, -1)
                mask = mask.any(dim=-1)

        return EmbedderOutput(embeddings=embeddings, mask=mask)

    def get_output_dim(self) -> int:
        return self._embedding.embedding_dim


@BaseEmbedder.register("analyzed_text")
class AnalyzedTextEmbedder(BaseEmbedder[IAnalyzedTextBatch]):
    """Embedder for analyzed text with multiple linguistic features.

    This embedder combines embeddings from multiple linguistic representations
    (surface forms, part-of-speech tags, character sequences) by concatenating
    them along the feature dimension.

    Args:
        surface: Optional embedder for surface form tokens.
        postag: Optional embedder for part-of-speech tags.
        character: Optional embedder for character sequences.

    Raises:
        ValueError: If all embedders are None (at least one is required).

    Example:
        >>> from formed.integrations.torch.modules import (
        ...     AnalyzedTextEmbedder,
        ...     TokenEmbedder
        ... )
        >>>
        >>> embedder = AnalyzedTextEmbedder(
        ...     surface=TokenEmbedder(vocab_size=10000, embedding_dim=128),
        ...     postag=TokenEmbedder(vocab_size=50, embedding_dim=32),
        ...     character=TokenEmbedder(vocab_size=256, embedding_dim=32)
        ... )
        >>>
        >>> # Output dimension is sum of all embedding dimensions (128 + 32 + 32 = 192)
        >>> assert embedder.get_output_dim() == 192

    Note:
        All provided embedders share the same mask, which is taken from
        the last non-None embedder processed.

    """

    def __init__(
        self,
        surface: BaseEmbedder[IIDSequenceBatch] | None = None,
        postag: BaseEmbedder[IIDSequenceBatch] | None = None,
        character: BaseEmbedder[IIDSequenceBatch] | None = None,
    ) -> None:
        super().__init__()
        if all(embedder is None for embedder in (surface, postag, character)):
            raise ValueError("At least one embedder must be provided for AnalyzedTextEmbedder.")

        self._surface = surface
        self._postag = postag
        self._character = character

    def forward(self, inputs: IAnalyzedTextBatch) -> EmbedderOutput:
        embeddings: list[torch.Tensor] = []
        mask: torch.Tensor | None = None

        for embedder, ids in (
            (self._surface, inputs.surfaces),
            (self._postag, inputs.postags),
            (self._character, inputs.characters),
        ):
            if embedder is not None and ids is not None:
                output = embedder(ids)
                embeddings.append(output.embeddings)
                mask = output.mask

        if not embeddings:
            raise ValueError("No embeddings were computed in AnalyzedTextEmbedder.")
        assert mask is not None

        return EmbedderOutput(embeddings=torch.cat(embeddings, dim=-1), mask=mask)

    def get_output_dim(self) -> int:
        return sum(
            embedder.get_output_dim()
            for embedder in (self._surface, self._postag, self._character)
            if embedder is not None
        )
