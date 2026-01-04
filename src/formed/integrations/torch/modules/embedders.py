"""Text embedding modules for PyTorch models.

This module provides embedders that convert tokenized text into dense vector
representations. Embedders handle various text representations including
surface forms, part-of-speech tags, and character sequences.

Key Components:
    - `BaseEmbedder`: Abstract base class for all embedders
    - `TokenEmbedder`: Embeds token ID sequences into dense vectors
    - `AnalyzedTextEmbedder`: Combines multiple embedding types (surface, POS, chars)

Features:
    - Support for nested token sequences (e.g., word -> character)
    - Automatic masking and padding handling
    - Configurable vectorization for character-level embeddings
    - Concatenation of multiple embedding types

Examples:
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
from collections.abc import Callable
from contextlib import suppress
from os import PathLike
from typing import TYPE_CHECKING, Any, Generic, Literal, NamedTuple, Optional, Protocol, Union, runtime_checkable

import torch
import torch.nn as nn
from colt import Registrable
from typing_extensions import Self, TypeVar

from ..initializers import BaseTensorInitializer
from ..types import IIDSequenceBatch, TensorCompatible, TensorCompatibleT
from ..utils import ensure_torch_tensor
from .scalarmix import ScalarMix
from .vectorizers import BaseSequenceVectorizer

if TYPE_CHECKING:
    with suppress(ImportError):
        from transformers import PreTrainedModel
        from transformers.models.auto.auto_factory import _BaseAutoModelClass

_TextBatchT = TypeVar("_TextBatchT")


@runtime_checkable
class IVariableTensorBatch(Protocol[TensorCompatibleT]):
    """Protocol for variable-length tensor batches.

    Attributes:
        tensor: Tensor of shape `(batch_size, seq_len, feature_dim)`.
        mask: Attention mask of shape `(batch_size, seq_len)`.

    """

    tensor: TensorCompatibleT
    mask: TensorCompatibleT

    def __len__(self) -> int: ...


SurfaceBatchT = TypeVar("SurfaceBatchT", bound="IIDSequenceBatch", default=Any)
PostagBatchT = TypeVar("PostagBatchT", bound=Union["IIDSequenceBatch", None], default=Any)
CharacterBatchT = TypeVar("CharacterBatchT", bound=Union["IIDSequenceBatch", None], default=Any)
TokenVectorBatchT = TypeVar("TokenVectorBatchT", bound=Union["IVariableTensorBatch", None], default=Any)


@runtime_checkable
class IAnalyzedTextBatch(
    Protocol[
        SurfaceBatchT,
        PostagBatchT,
        CharacterBatchT,
        TokenVectorBatchT,
    ]
):
    """Protocol for analyzed text batches with multiple linguistic features.

    Attributes:
        surfaces: Surface form token IDs.
        postags: Part-of-speech tag IDs (optional).
        characters: Character sequence IDs (optional).
        token_vectors: Token-level dense vectors (optional).

    """

    surfaces: SurfaceBatchT
    postags: PostagBatchT
    characters: CharacterBatchT
    token_vectors: TokenVectorBatchT


class EmbedderOutput(NamedTuple):
    """Output from an embedder.

    Attributes:
        embeddings: Dense embeddings of shape `(batch_size, seq_len, embedding_dim)`.
        mask: Attention mask of shape `(batch_size, seq_len)`.

    """

    embeddings: torch.Tensor
    mask: torch.Tensor


class BaseEmbedder(nn.Module, Registrable, Generic[_TextBatchT], abc.ABC):
    """Abstract base class for text embedders.

    Embedders convert tokenized text into dense vector representations.
    They output both embeddings and attention masks.

    Type Parameters:
        _TextBatchT: Type of input batch (e.g., `IIDSequenceBatch`, `IAnalyzedTextBatch`).

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


@BaseEmbedder.register("pass_through")
class PassThroughEmbedder(BaseEmbedder[IVariableTensorBatch[TensorCompatibleT]]):
    """Embedder that passes through input tensors unchanged.

    This embedder is useful when the input tensors are already in the desired
    embedding format. It simply returns the input tensors and their masks.

    Examples:
        >>> from formed.integrations.torch.modules import PassThroughEmbedder
        >>>
        >>> embedder = PassThroughEmbedder()
        >>> output = embedder(variable_tensor_batch)
        >>> assert torch.equal(output.embeddings, variable_tensor_batch.tensor)
        >>> assert torch.equal(output.mask, variable_tensor_batch.mask)

    """

    def forward(
        self,
        inputs: IVariableTensorBatch[TensorCompatibleT],
    ) -> EmbedderOutput:
        tensor = ensure_torch_tensor(inputs.tensor)
        mask = ensure_torch_tensor(inputs.mask).bool()
        return EmbedderOutput(embeddings=tensor, mask=mask)

    def get_output_dim(self) -> int:
        raise NotImplementedError("PassThroughEmbedder does not have a fixed output dimension.")


@BaseEmbedder.register("token")
class TokenEmbedder(BaseEmbedder["IIDSequenceBatch"]):
    """Embedder for token ID sequences.

    This embedder converts token IDs into dense embeddings using a learned
    embedding matrix. It supports both 2D `(batch_size, seq_len)` and 3D
    `(batch_size, seq_len, char_len)` token ID tensors.

    For 3D inputs (e.g., character-level tokens within words), the embedder
    can either average the embeddings or apply a custom vectorizer.

    Args:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of the embedding vectors.
        padding_idx: Index of the padding token (default: `0`).
        vectorizer: Optional vectorizer for 3D inputs (character sequences).

    Examples:
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
        initializer: BaseTensorInitializer | Callable[[], TensorCompatible],
        *,
        padding_idx: int = 0,
        freeze: bool = False,
        vectorizer: Optional[BaseSequenceVectorizer] = None,
    ) -> None:
        weight = ensure_torch_tensor(initializer())

        super().__init__()
        self._embedding = nn.Embedding.from_pretrained(weight, padding_idx=padding_idx, freeze=freeze)
        self._vectorizer = vectorizer

    def forward(self, inputs: "IIDSequenceBatch") -> EmbedderOutput:
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


@BaseEmbedder.register("pretrained_transformer")
class PretrainedTransformerEmbedder(BaseEmbedder[IIDSequenceBatch]):
    """Embedder using pretrained transformer models from Hugging Face.

    This embedder wraps pretrained transformer models (BERT, RoBERTa, etc.)
    to extract contextualized embeddings. It uses the last hidden state from
    the transformer as the embedding representation.

    Args:
        model: Either a model name/path string, `PathLike` object, or a `PreTrainedModel` instance.
            If a string or `PathLike`, the model will be loaded using transformers auto classes.
        auto_class: The auto class to use for loading the model.
        subcmodule: Optional submodule path to extract from the loaded model (e.g., `"encoder"`).
        freeze: If `True`, freezes all model parameters (no gradient computation).
        **kwargs: Additional keyword arguments passed to the model loader.

    Examples:
        >>> # Load a pretrained BERT model
        >>> embedder = PretrainedTransformerEmbedder(
        ...     model="bert-base-uncased",
        ...     freeze=True
        ... )
        >>>
        >>> # Use a specific auto class
        >>> from transformers import AutoModel
        >>> embedder = PretrainedTransformerEmbedder(
        ...     model="roberta-base",
        ...     auto_class=AutoModel,
        ...     freeze=False
        ... )
        >>>
        >>> # Use an already loaded model
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained("bert-base-uncased")
        >>> embedder = PretrainedTransformerEmbedder(model=model)

    Note:
        Models are cached using LRU cache by the `load_pretrained_transformer` utility.
        When `freeze=True`, all model parameters have `requires_grad=False`.

    """

    class _Embedding(torch.nn.Module):
        def __init__(self, model: "PreTrainedModel") -> None:
            super().__init__()
            self.embedding = model.get_input_embeddings()

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embedding(input_ids)

        def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
            return super().__call__(input_ids)

    def __init__(
        self,
        model: Union[str, PathLike, "PreTrainedModel"],
        auto_class: str | type["_BaseAutoModelClass"] | None = None,
        subcmodule: str | None = None,
        freeze: bool = False,
        eval_mode: bool = False,
        layer_to_use: Literal["embeddings", "last", "all"] = "last",
        gradient_checkpointing: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(model, (str, PathLike)):
            from formed.integrations.transformers import load_pretrained_transformer

            model = load_pretrained_transformer.__wrapped__(
                model,
                auto_class=auto_class,
                submodule=subcmodule,
                **kwargs,
            )

        super().__init__()

        self._model = model
        self._scalar_mix: ScalarMix | None = None
        self._eval_mode = eval_mode
        self._output_dim = model.config.hidden_size
        self._vocab_size = model.config.vocab_size

        if gradient_checkpointing is not None:
            self._model.config.update({"gradient_checkpointing": gradient_checkpointing})

        if self._eval_mode:
            self._model.eval()

        if freeze:
            for param in self._model.parameters():
                param.requires_grad = False

        if layer_to_use == "all":
            self._scalar_mix = ScalarMix(self._model.config.num_hidden_layers)
            self._model.config.output_hidden_states = True
        elif layer_to_use == "embeddings":
            self._model = PretrainedTransformerEmbedder._Embedding(self._model)

    def forward(self, inputs: IIDSequenceBatch) -> EmbedderOutput:
        input_ids = ensure_torch_tensor(inputs.ids)
        mask = ensure_torch_tensor(inputs.mask)

        if isinstance(self._model, PretrainedTransformerEmbedder._Embedding):
            embeddings = self._model(input_ids)
        else:
            transformer_outputs = self._model(input_ids=input_ids, attention_mask=mask)
            if self._scalar_mix is not None:
                # The hidden states will also include the embedding layer, which we don't
                # include in the scalar mix. Hence the `[1:]` slicing.
                hidden_states = transformer_outputs.hidden_states[1:]
                embeddings = self._scalar_mix(hidden_states)
            else:
                embeddings = transformer_outputs.last_hidden_state
        return EmbedderOutput(embeddings, mask)

    def get_output_dim(self) -> int:
        return self._output_dim

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def train(self, mode: bool = True) -> Self:
        self.training = mode
        for name, module in self.named_children():
            if self._eval_mode and name == "_model":
                module.eval()
            else:
                module.train(mode)
        return self


@BaseEmbedder.register("analyzed_text")
class AnalyzedTextEmbedder(BaseEmbedder["IAnalyzedTextBatch"]):
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

    Examples:
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
        surface: Optional["BaseEmbedder[IIDSequenceBatch]"] = None,
        postag: Optional["BaseEmbedder[IIDSequenceBatch]"] = None,
        character: Optional["BaseEmbedder[IIDSequenceBatch]"] = None,
        token_vector: Optional["BaseEmbedder[IVariableTensorBatch]"] = None,
    ) -> None:
        super().__init__()
        if all(embedder is None for embedder in (surface, postag, character)):
            raise ValueError("At least one embedder must be provided for AnalyzedTextEmbedder.")

        self._surface = surface
        self._postag = postag
        self._character = character
        self._token_vector = token_vector

    def forward(self, inputs: "IAnalyzedTextBatch") -> EmbedderOutput:
        embeddings: list[torch.Tensor] = []
        mask: Optional[torch.Tensor] = None

        for embedder, ids in (
            (self._surface, inputs.surfaces),
            (self._postag, inputs.postags),
            (self._character, inputs.characters),
        ):
            if embedder is not None and ids is not None:
                output = embedder(ids)
                embeddings.append(output.embeddings)
                mask = output.mask

        if self._token_vector is not None and inputs.token_vectors is not None:
            output = self._token_vector(inputs.token_vectors)
            embeddings.append(output.embeddings)

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
