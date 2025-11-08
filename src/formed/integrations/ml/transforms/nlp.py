import dataclasses
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from logging import getLogger
from typing import Any, Generic, Optional, Union, cast

import numpy
from typing_extensions import TypeVar

from formed.common.nlputils import punkt_tokenize

from ..types import AnalyzedText, AsBatch, AsConverter, AsInstance, DataModuleModeT, IDSequenceBatch  # noqa: F401
from .base import BaseTransform, DataModule, Extra, Param

logger = getLogger(__name__)


_S = TypeVar("_S", default=Any)


@BaseTransform.register("tokens")
class TokenSequenceIndexer(
    BaseTransform[_S, Sequence[str], Sequence[str], IDSequenceBatch],
    Generic[_S],
):
    vocab: Mapping[str, int] = dataclasses.field(default_factory=dict)
    pad_token: str = "<PAD>"
    unk_token: Optional[str] = None
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    min_df: Union[int, float] = 1
    max_df: Union[int, float] = 1.0
    max_vocab_size: Optional[int] = None
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
        return self.vocab[self.pad_token]

    @property
    def unk_index(self) -> Optional[int]:
        if self.unk_token is not None:
            return self.vocab[self.unk_token]
        return None

    @property
    def bos_index(self) -> Optional[int]:
        if self.bos_token is not None:
            return self.vocab[self.bos_token]
        return None

    @property
    def eos_index(self) -> Optional[int]:
        if self.eos_token is not None:
            return self.vocab[self.eos_token]
        return None

    @property
    def vocab_size(self) -> int:
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
        if value in self.vocab:
            return self.vocab[value]
        if self.unk_token is not None:
            return self.vocab[self.unk_token]
        raise KeyError(value)

    def get_value(self, index: int, /) -> str:
        if index in self._inverted_vocab:
            return self._inverted_vocab[index]
        raise KeyError(index)

    def ingest(self, values: Sequence[str], /) -> None:
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
    min_characters: int = 1

    def _get_input_value(self, data: _S) -> Optional[Sequence[str]]:
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
    surfaces: TokenSequenceIndexer = dataclasses.field(default_factory=TokenSequenceIndexer)
    postags: Extra[TokenSequenceIndexer] = Extra.default(None)
    characters: Extra[TokenCharactersIndexer] = Extra.default(None)

    analyzer: Param[Optional[Callable[[Union[str, Sequence[str], AnalyzedText]], AnalyzedText]]] = Param.default(None)

    @staticmethod
    def _default_analyzer(text: Union[str, Sequence[str], AnalyzedText]) -> AnalyzedText:
        if isinstance(text, AnalyzedText):
            return text
        surfaces = punkt_tokenize(text) if isinstance(text, str) else text
        return AnalyzedText(surfaces=surfaces)

    def instance(
        self: "Tokenizer[AsConverter]", x: Union[str, Sequence[str], AnalyzedText], /
    ) -> "Tokenizer[AsInstance]":
        analyzer = self.analyzer or self._default_analyzer
        return cast(DataModule[AsConverter], super()).instance(analyzer(x))
