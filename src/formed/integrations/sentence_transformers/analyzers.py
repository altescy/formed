import dataclasses
import unicodedata
from functools import cached_property
from os import PathLike
from typing import Literal

from formed.integrations.ml.types import AnalyzedText

from .utils import load_sentence_transformer


@dataclasses.dataclass
class SentenceTransformerAnalyzer:
    model_name_or_path: str | PathLike
    unicode_normalization: Literal["NFC", "NFKC", "NFD", "NFKD"] | None = None

    @cached_property
    def _tokenizer(self):
        return load_sentence_transformer(self.model_name_or_path).tokenizer

    def __call__(self, text: str) -> AnalyzedText:
        if self.unicode_normalization:
            text = unicodedata.normalize(self.unicode_normalization, text)
        if self._tokenizer.__module__.startswith("tokenizers"):
            tokens = self._tokenizer.encode(text).tokens
        else:
            tokens = self._tokenizer.tokenize(text)
        return AnalyzedText(surfaces=tokens)
