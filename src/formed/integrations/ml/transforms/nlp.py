"""NLP-specific data transformations for text processing.

This module provides transformations for natural language processing tasks,
including tokenization, vocabulary building, and sequence indexing with special
tokens (PAD, UNK, BOS, EOS).

Available Transforms:
    - `TokenSequenceIndexer`: Convert token sequences to integer indices with vocab building
    - `TokenCharactersIndexer`: Character-level indexing for tokens
    - `Tokenizer`: Complete tokenization pipeline with surfaces, postags, and characters

Features:
    - Dynamic vocabulary building with `min_df`/`max_df` filtering
    - Document frequency tracking
    - Special token handling (PAD, UNK, BOS, EOS)
    - Automatic padding and masking
    - Reconstruction support (indices -> tokens)

Examples:
    >>> from formed.integrations.ml import Tokenizer, TokenSequenceIndexer
    >>>
    >>> # Simple tokenization
    >>> tokenizer = Tokenizer(surfaces=TokenSequenceIndexer(
    ...     unk_token="<UNK>", pad_token="<PAD>"
    ... ))
    >>>
    >>> with tokenizer.train():
    ...     instance = tokenizer.instance("Hello world!")
    >>> batch = tokenizer.batch([instance1, instance2, instance3])
    >>> print(batch.surfaces.ids.shape)  # (3, max_length)
    >>> print(batch.surfaces.mask.shape)  # (3, max_length)
    >>>
    >>> # Reconstruct tokens from indices
    >>> tokens = tokenizer.surfaces.reconstruct(batch.surfaces)

"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from logging import getLogger
from typing import Any, Generic, Union, cast

import numpy
from typing_extensions import TypeVar

from formed.common.nlputils import punkt_tokenize

from ..types import AnalyzedText, AsBatch, AsConverter, AsInstance, DataModuleModeT, IDSequenceBatch  # noqa: F401
from .base import BaseTransform, DataModule, Extra, Param
from .basic import TensorSequenceTransform, TensorTransform

logger = getLogger(__name__)


_S = TypeVar("_S", default=Any)


@BaseTransform.register("tokens")
class TokenSequenceIndexer(
    BaseTransform[_S, Sequence[str], Sequence[str], IDSequenceBatch],
    Generic[_S],
):
    """Convert token sequences to integer indices with vocabulary building and filtering.

    TokenSequenceIndexer builds and maintains a vocabulary, converting tokens to indices.
    It supports vocabulary filtering by document frequency (`min_df`/`max_df`), vocabulary
    size limits, and special tokens (PAD, UNK, BOS, EOS). During batching, sequences
    are padded to the same length and a mask is generated.

    Type Parameters:
        _S: Source data type before accessor

    Attributes:
        vocab: Pre-defined or built vocabulary mapping tokens to indices.
        pad_token: Padding token (required).
        unk_token: Unknown token for out-of-vocabulary tokens (optional).
        bos_token: Beginning-of-sequence token (optional).
        eos_token: End-of-sequence token (optional).
        min_df: Minimum document frequency (int or fraction) to include token.
        max_df: Maximum document frequency (int or fraction) to include token.
        max_vocab_size: Maximum vocabulary size (excluding special tokens).
        freeze: If True, prevent vocabulary updates.


    Examples:
        >>> # Build vocabulary with filtering
        >>> indexer = TokenSequenceIndexer(
        ...     unk_token="<UNK>",
        ...     min_df=2,  # Tokens must appear in at least 2 documents
        ...     max_vocab_size=10000
        ... )
        >>>
        >>> with indexer.train():
        ...     tokens1 = indexer.instance(["hello", "world"])
        ...     tokens2 = indexer.instance(["hello", "there"])
        >>> # Vocabulary: {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3, "there": 4}
        >>>
        >>> # Create batch with padding
        >>> batch = indexer.batch([
        ...     ["hello", "world"],
        ...     ["hello", "there", "friend"]
        ... ])
        >>> print(batch.ids.shape)  # (2, 3) - padded to max length
        >>> print(batch.mask.shape)  # (2, 3) - True for real tokens
        >>>
        >>> # Reconstruct original tokens
        >>> tokens = indexer.reconstruct(batch)
        >>> print(tokens)  # [["hello", "world"], ["hello", "there", "friend"]]

    Note:
        - Special tokens are always added first and never filtered
        - `min_df`/`max_df` require `unk_token` to handle filtered tokens
        - Document frequency counts unique tokens per document
        - BOS/EOS tokens are added during batching if specified
        - Reconstruction removes special tokens and padding

    Raises:
        ValueError: If configuration is invalid (e.g., min_df>1 without unk_token).

    """

    vocab: Mapping[str, int] = dataclasses.field(default_factory=dict)
    pad_token: str = "<PAD>"
    unk_token: str | None = None
    bos_token: str | None = None
    eos_token: str | None = None
    min_df: int | float = 1
    max_df: int | float = 1.0
    max_vocab_size: int | None = None
    freeze: bool = False

    _token_counts: dict[str, int] = dataclasses.field(default_factory=dict, init=False, repr=False)
    _document_count: int = dataclasses.field(default=0, init=False, repr=False)
    _document_frequencies: dict[str, int] = dataclasses.field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.min_df < 0 or (isinstance(self.min_df, float) and not (0.0 < self.min_df <= 1.0)):
            raise ValueError("min_df must be a non-negative integer or a float in (0.0, 1.0]")
        if self.max_df < 0 or (isinstance(self.max_df, float) and not (0.0 < self.max_df <= 1.0)):
            raise ValueError("max_df must be a non-negative integer or a float in (0.0, 1.0]")
        if self.unk_token is None and (
            (isinstance(self.min_df, int) and self.min_df > 1)
            or (isinstance(self.min_df, float) and self.min_df > 0.0)
            or (isinstance(self.max_df, int) and self.max_df < float("inf"))
            or (isinstance(self.max_df, float) and self.max_df < 1.0)
            or self.max_vocab_size is not None
        ):
            raise ValueError("unk_token must be specified if min_df > 1, max_df < inf, or max_vocab_size is set")

        special_tokens = [
            token for token in (self.pad_token, self.unk_token, self.bos_token, self.eos_token) if token is not None
        ]
        special_token_set = set(special_tokens)
        if len(special_tokens) != len(special_token_set):
            raise ValueError("Special tokens must be unique")

        vocab = dict(self.vocab or {})
        if self.freeze:
            if not all(token in self.vocab for token in special_token_set):
                raise ValueError("All special tokens must be in the vocab when freeze is True")
        else:
            for token in special_tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        self.vocab: Mapping[str, int] = vocab

        if not all(index in self._inverted_vocab for index in range(len(self.vocab))):
            raise ValueError("Vocab indices must be contiguous")

    @cached_property
    def _inverted_vocab(self) -> Mapping[int, str]:
        inverted_vocab = {idx: token for token, idx in self.vocab.items()}
        if not all(index in inverted_vocab for index in range(len(self.vocab))):
            raise ValueError("Vocab indices must be contiguous")
        return inverted_vocab

    @property
    def pad_index(self) -> int:
        """Index of the padding token."""
        return self.vocab[self.pad_token]

    @property
    def unk_index(self) -> int | None:
        """Index of the unknown token, or `None` if not set."""
        if self.unk_token is not None:
            return self.vocab[self.unk_token]
        return None

    @property
    def bos_index(self) -> int | None:
        """Index of the beginning-of-sequence token, or `None` if not set."""
        if self.bos_token is not None:
            return self.vocab[self.bos_token]
        return None

    @property
    def eos_index(self) -> int | None:
        """Index of the end-of-sequence token, or `None` if not set."""
        if self.eos_token is not None:
            return self.vocab[self.eos_token]
        return None

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.vocab)

    def _on_start_training(self) -> None:
        self._token_counts.clear()
        self._document_frequencies.clear()

    def _on_end_training(self) -> None:
        if self.freeze:
            return
        num_documents = self._document_count
        min_df = self.min_df if isinstance(self.min_df, int) else int(self.min_df * num_documents)
        max_df = self.max_df if isinstance(self.max_df, int) else int(self.max_df * num_documents)

        tokens = [token for token, df in self._document_frequencies.items() if min_df <= df <= max_df]
        if self.max_vocab_size is not None:
            tokens.sort(key=lambda token: self._token_counts[token], reverse=True)
            tokens = tokens[: self.max_vocab_size]

        vocab = dict(self.vocab)
        for token in sorted(tokens):
            if token not in vocab:
                vocab[token] = len(vocab)

        self.vocab = vocab
        del self._inverted_vocab
        self._inverted_vocab  # Recompute inverted vocab

    def get_index(self, value: str, /) -> int:
        """Get the index of a token, using unk_token if not found."""
        if value in self.vocab:
            return self.vocab[value]
        if self.unk_token is not None:
            return self.vocab[self.unk_token]
        raise KeyError(value)

    def get_value(self, index: int, /) -> str:
        """Get the token corresponding to an index."""
        if index in self._inverted_vocab:
            return self._inverted_vocab[index]
        raise KeyError(index)

    def ingest(self, values: Sequence[str], /) -> None:
        """Ingest a sequence of tokens to update counts for vocabulary building."""
        if self.freeze:
            return
        if self._training:
            for token in values:
                self._token_counts[token] = self._token_counts.get(token, 0) + 1
            self._document_count += 1
            for toke in set(values):
                self._document_frequencies[toke] = self._document_frequencies.get(toke, 0) + 1
        else:
            logger.warning("Ignoring ingest call when not in training mode")

    def instance(self, tokens: Sequence[str], /) -> Sequence[str]:
        if self._training:
            self.ingest(tokens)
        return tokens

    def batch(self, batch: Sequence[Sequence[str]], /) -> IDSequenceBatch:
        batch_size = len(batch)
        max_length = max(len(tokens) for tokens in batch)
        if self.bos_token is not None:
            max_length += 1
        if self.eos_token is not None:
            max_length += 1
        ids = numpy.full((batch_size, max_length), self.pad_index, dtype=numpy.int64)
        mask = numpy.zeros((batch_size, max_length), dtype=numpy.bool_)
        for i, tokens in enumerate(batch):
            indices = [self.get_index(token) for token in tokens]
            if self.bos_token is not None:
                indices = [self.vocab[self.bos_token]] + indices
            if self.eos_token is not None:
                indices = indices + [self.vocab[self.eos_token]]
            length = len(indices)
            ids[i, :length] = indices
            mask[i, :length] = 1
        return IDSequenceBatch(ids=ids, mask=mask)

    def reconstruct(self, batch: IDSequenceBatch, /) -> list[Sequence[str]]:
        """Reconstruct token sequences from a batch of indices."""
        sequences = []
        for i in range(batch.ids.shape[0]):
            length = int(batch.mask[i].sum())
            indices = batch.ids[i, :length].tolist()
            tokens = [self.get_value(index) for index in indices]
            if tokens and tokens[0] == self.bos_token:
                tokens = tokens[1:]
            if tokens and tokens[-1] == self.eos_token:
                tokens = tokens[:-1]
            tokens = [token for token in tokens if token != self.pad_token]
            sequences.append(tokens)
        return sequences


@BaseTransform.register("token_characters")
class TokenCharactersIndexer(TokenSequenceIndexer[_S], Generic[_S]):
    """Character-level indexing for token sequences.

    TokenCharactersIndexer extends TokenSequenceIndexer to index individual characters
    within tokens. This is useful for character-level models or handling rare words.
    The batch output is a 3D tensor: (batch_size, num_tokens, max_characters).

    Type Parameters:
        _S: Source data type before accessor

    Attributes:
        min_characters: Minimum character length per token (for padding).

    Note:
        Inherits all attributes from TokenSequenceIndexer.

    Examples:
        >>> indexer = TokenCharactersIndexer(
        ...     unk_token="<UNK>",
        ...     bos_token="<BOS>",
        ...     eos_token="<EOS>"
        ... )
        >>>
        >>> with indexer.train():
        ...     tokens = indexer.instance(["hello", "world"])
        >>>
        >>> batch = indexer.batch([["hello", "world"], ["hi"]])
        >>> print(batch.ids.shape)  # (2, 2, 7) - batch x tokens x chars
        >>> print(batch.mask.shape)  # (2, 2, 7)

    Note:
        - Each token is converted to a sequence of character indices
        - BOS/EOS are added per token, not per sequence
        - Vocabulary contains individual characters, not tokens
        - Useful for morphologically rich languages or rare word handling

    """

    min_characters: int = 1

    def _get_input_value(self, data: _S) -> Sequence[str] | None:
        if isinstance(data, AnalyzedText) and self.accessor is None:
            return data.surfaces
        return super()._get_input_value(data)

    def ingest(self, values: Sequence[str], /) -> None:
        super().ingest("".join(values))

    def batch(self, batch: Sequence[Sequence[str]], /) -> IDSequenceBatch:
        batch_size = len(batch)
        max_tokens = max(len(tokens) for tokens in batch)
        max_characters = max(self.min_characters, max(len(token) for tokens in batch for token in tokens))
        ids = numpy.full((batch_size, max_tokens, max_characters), self.pad_index, dtype=numpy.int64)
        mask = numpy.zeros((batch_size, max_tokens, max_characters), dtype=numpy.bool_)
        for i, tokens in enumerate(batch):
            for j, token in enumerate(tokens):
                indices = [self.get_index(char) for char in token]
                if self.bos_token is not None:
                    indices = [self.vocab[self.bos_token]] + indices
                if self.eos_token is not None:
                    indices = indices + [self.vocab[self.eos_token]]
                length = len(indices)
                ids[i, j, :length] = indices
                mask[i, j, :length] = 1
        return IDSequenceBatch(ids=ids, mask=mask)

    def reconstruct(self, batch: IDSequenceBatch, /) -> list[Sequence[str]]:
        sequences = []
        for i in range(batch.ids.shape[0]):
            token_indices = batch.ids[i]
            token_mask = batch.mask[i]
            tokens = []
            for j in range(token_indices.shape[0]):
                if not token_mask[j].any():
                    break
                length = int(token_mask[j].sum())
                char_indices = token_indices[j, :length].tolist()
                chars = [self.get_value(index) for index in char_indices]
                tokens.append("".join(chars))
            if tokens and tokens[0] == self.bos_token:
                tokens = tokens[1:]
            if tokens and tokens[-1] == self.eos_token:
                tokens = tokens[:-1]
            tokens = [token for token in tokens if token != self.pad_token]
            sequences.append(tokens)
        return sequences


@BaseTransform.register("tokenizer")
class Tokenizer(
    DataModule[
        DataModuleModeT,
        Union[str, Sequence[str], AnalyzedText],
        "Tokenizer[AsInstance]",
        "Tokenizer[AsBatch]",
    ],
    Generic[DataModuleModeT],
):
    """Complete tokenization pipeline with multiple representation options.

    Tokenizer is a DataModule that provides a unified interface for text tokenization
    with support for surface forms, part-of-speech tags, and character-level representations.
    It accepts raw text, pre-tokenized sequences, or analyzed text and converts them
    to indexed representations suitable for neural models.

    Type Parameters:
        DataModuleModeT: Current mode (AsConverter, AsInstance, or AsBatch)

    Attributes:
        surfaces: Required token sequence indexer for surface forms (words).
        postags: Optional indexer for part-of-speech tags.
        characters: Optional character-level indexer for tokens.
        analyzer: Optional custom text analyzer/tokenizer function.

    Examples:
        >>> # Basic tokenization
        >>> tokenizer = Tokenizer(
        ...     surfaces=TokenSequenceIndexer(unk_token="<UNK>")
        ... )
        >>>
        >>> with tokenizer.train():
        ...     instance1 = tokenizer.instance("Hello world!")
        ...     instance2 = tokenizer.instance(["Hello", "world", "!"])
        >>>
        >>> batch = tokenizer.batch([instance1, instance2])
        >>> print(batch.surfaces.ids.shape)  # (2, max_tokens)
        >>> print(batch.surfaces.mask.shape)  # (2, max_tokens)
        >>>
        >>> # With POS tags and characters
        >>> tokenizer = Tokenizer(
        ...     surfaces=TokenSequenceIndexer(unk_token="<UNK>"),
        ...     postags=TokenSequenceIndexer(unk_token="<UNK-POS>"),
        ...     characters=TokenCharactersIndexer()
        ... )
        >>>
        >>> analyzed = AnalyzedText(
        ...     surfaces=["Hello", "world"],
        ...     postags=["INTJ", "NOUN"]
        ... )
        >>> instance = tokenizer.instance(analyzed)
        >>> print(instance.surfaces)  # Indexed tokens
        >>> print(instance.postags)  # Indexed POS tags
        >>> print(instance.characters)  # Character indices

    Note:
        - Default analyzer uses punkt tokenization for raw strings
        - Accepts string, token list, or AnalyzedText as input
        - Extra fields (postags, characters) can be None
        - All indexers share the same training context

    """

    surfaces: TokenSequenceIndexer = dataclasses.field(default_factory=TokenSequenceIndexer)
    postags: Extra[TokenSequenceIndexer] = Extra.default(None)
    characters: Extra[TokenCharactersIndexer] = Extra.default(None)
    text_vector: Extra[TensorTransform] = Extra.default(None)
    token_vectors: Extra[TensorSequenceTransform] = Extra.default(None)

    analyzer: Param[Callable[[str | Sequence[str] | AnalyzedText], AnalyzedText] | None] = Param.default(None)

    @staticmethod
    def _default_analyzer(text: str | Sequence[str] | AnalyzedText) -> AnalyzedText:
        if isinstance(text, AnalyzedText):
            return text
        surfaces = punkt_tokenize(text) if isinstance(text, str) else text
        return AnalyzedText(surfaces=surfaces)

    def instance(self: "Tokenizer[AsConverter]", x: str | Sequence[str] | AnalyzedText, /) -> "Tokenizer[AsInstance]":
        analyzer = self.analyzer or self._default_analyzer
        return cast(DataModule[AsConverter], super()).instance(analyzer(x))
