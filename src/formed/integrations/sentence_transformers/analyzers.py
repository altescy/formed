import dataclasses
import unicodedata
from collections.abc import Sequence
from functools import cached_property
from os import PathLike
from typing import Literal

from formed.integrations.ml import BaseTextAnalyzer
from formed.integrations.ml.types import AnalyzedText

from .utils import load_sentence_transformer


@dataclasses.dataclass
class SentenceTransformerAnalyzer(BaseTextAnalyzer):
    model_name_or_path: str | PathLike
    unicode_normalization: Literal["NFC", "NFKC", "NFD", "NFKD"] | None = None

    @cached_property
    def _tokenizer(self):
        return load_sentence_transformer(self.model_name_or_path).tokenizer

    def __call__(self, text: str | Sequence[str] | AnalyzedText) -> AnalyzedText:
        return super().__call__(text)

    def tokenize(self, text: str) -> list[str]:
        if self.unicode_normalization:
            text = unicodedata.normalize(self.unicode_normalization, text)
        if self._tokenizer.__module__.startswith("tokenizers"):
            return self._tokenizer.encode(text).tokens
        return self._tokenizer.tokenize(text)

    def detokenize(self, tokens: Sequence[str]) -> str:
        if hasattr(self._tokenizer, "convert_tokens_to_string"):
            return self._tokenizer.convert_tokens_to_string(list(tokens))
        return " ".join(tokens)
