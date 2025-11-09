from typing import Generic, NamedTuple, Optional, TypeVar

import jax
from colt import Registrable
from flax import nnx

from ..types import IAnalyzedTextBatch, IIDSequenceBatch
from ..utils import ensure_jax_array
from .vectorizers import BaseSequenceVectorizer

_TextBatchT = TypeVar("_TextBatchT")


class EmbedderOutput(NamedTuple):
    embeddings: jax.Array
    mask: jax.Array


class BaseEmbedder(nnx.Module, Registrable, Generic[_TextBatchT]):
    def __call__(self, inputs: _TextBatchT) -> EmbedderOutput:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


@BaseEmbedder.register("token")
class TokenEmbedder(BaseEmbedder[IIDSequenceBatch]):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        vectorizer: Optional[BaseSequenceVectorizer] = None,
        rngs: nnx.Rngs,
    ) -> None:
        self._embedding = nnx.Embed(num_embeddings=vocab_size, features=embedding_dim, rngs=rngs)
        self._vectorizer = vectorizer

    def __call__(self, inputs: IIDSequenceBatch) -> EmbedderOutput:
        nested = False
        if inputs.ids.ndim > 2:
            if inputs.ids.ndim != 3:
                raise ValueError("Token ids must be of shape (batch_size, seq_len) or (batch_size, seq_len, char_len)")
            nested = True

        if inputs.ids.shape != inputs.mask.shape:
            raise ValueError(
                f"Token ids and mask must have the same shape, got {inputs.ids.shape} and {inputs.mask.shape}"
            )

        token_ids = ensure_jax_array(inputs.ids)
        mask = ensure_jax_array(inputs.mask)

        embeddings = self._embedding(token_ids)

        if nested:
            if self._vectorizer is None:
                embeddings = (embeddings * mask[..., None]).sum(axis=-2) / mask.sum(axis=-1, keepdims=True).clip(min=1)
                mask = mask.any(axis=-1)
            else:
                embeddings = self._vectorizer(embeddings, mask=mask)
                mask = mask.any(axis=-1)

        return EmbedderOutput(embeddings=embeddings, mask=mask)

    def get_output_dim(self) -> int:
        return self._embedding.features


@BaseEmbedder.register("analyzed_text")
class AnalyzedTextEmbedder(BaseEmbedder[IAnalyzedTextBatch]):
    def __init__(
        self,
        surface: Optional[BaseEmbedder[IIDSequenceBatch]] = None,
        postag: Optional[BaseEmbedder[IIDSequenceBatch]] = None,
        character: Optional[BaseEmbedder[IIDSequenceBatch]] = None,
    ) -> None:
        if all(embedder is None for embedder in (surface, postag, character)):
            raise ValueError("At least one embedder must be provided for AnalyzedTextEmbedder.")

        self._surface = surface
        self._postag = postag
        self._character = character

    def __call__(self, inputs: IAnalyzedTextBatch) -> EmbedderOutput:
        embeddings: list[jax.Array] = []
        mask: Optional[jax.Array] = None

        for embedder, ids in (
            (self._surface, inputs.surfaces),
            (self._postag, inputs.postags),
            (self._character, inputs.characters),
        ):
            if embedder is not None:
                output = embedder(ids)
                embeddings.append(output.embeddings)
                mask = output.mask
        if not embeddings:
            raise ValueError("No embeddings were computed in AnalyzedTextEmbedder.")
        assert mask is not None
        return EmbedderOutput(embeddings=jax.numpy.concatenate(embeddings, axis=-1), mask=mask)

    def get_output_dim(self) -> int:
        return sum(
            embedder.get_output_dim()
            for embedder in (self._surface, self._postag, self._character)
            if embedder is not None
        )
