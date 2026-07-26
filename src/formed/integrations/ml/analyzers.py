import abc
import re
from collections.abc import Sequence
from typing import Literal

from colt import Registrable

from .types import AnalyzedText


class BaseTextAnalyzer(Registrable, abc.ABC):
    """Analyze text and provide a reversible surface-token representation."""

    def __call__(self, text: str | Sequence[str] | AnalyzedText) -> AnalyzedText:
        if isinstance(text, AnalyzedText):
            return text
        tokens = self.tokenize(text) if isinstance(text, str) else list(text)
        return AnalyzedText(surfaces=tokens)

    @abc.abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Convert raw text into surface tokens."""
        raise NotImplementedError

    @abc.abstractmethod
    def detokenize(self, tokens: Sequence[str]) -> str:
        """Convert surface tokens into their canonical text representation."""
        raise NotImplementedError


@BaseTextAnalyzer.register("punkt")
class PunktTextAnalyzer(BaseTextAnalyzer):
    """Tokenize around punctuation and whitespace with configurable preservation."""

    DEFAULT_DELIMITER_PATTERN = r"[,.!?@/\\'\"()\[\]{}-]|\s+"

    def __init__(
        self,
        delimiter_attachment: Literal["discard", "prefix", "suffix", "separate"] = "separate",
        delimiter_pattern: str = DEFAULT_DELIMITER_PATTERN,
        detokenization_separator: str = " ",
    ) -> None:
        self.delimiter_attachment = delimiter_attachment
        self.delimiter_pattern = delimiter_pattern
        self.detokenization_separator = detokenization_separator
        if delimiter_attachment not in ("discard", "prefix", "suffix", "separate"):
            raise ValueError(f"Unknown delimiter attachment mode: {delimiter_attachment}")
        self._delimiter_regex = re.compile(delimiter_pattern)
        if self._delimiter_regex.match("") is not None:
            raise ValueError("delimiter_pattern must not match an empty string")

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        pending_prefix = ""
        position = 0
        for match in self._delimiter_regex.finditer(text):
            if match.start() > position:
                token = text[position : match.start()]
                if pending_prefix:
                    token = pending_prefix + token
                    pending_prefix = ""
                tokens.append(token)

            delimiter = match.group()
            if self.delimiter_attachment == "separate":
                tokens.append(delimiter)
            elif self.delimiter_attachment == "suffix":
                if tokens:
                    tokens[-1] += delimiter
                else:
                    pending_prefix += delimiter
            elif self.delimiter_attachment == "prefix":
                pending_prefix += delimiter
            position = match.end()

        if position < len(text):
            token = text[position:]
            tokens.append(pending_prefix + token)
            pending_prefix = ""

        if pending_prefix:
            if tokens:
                tokens[-1] += pending_prefix
            else:
                tokens.append(pending_prefix)
        return tokens

    def detokenize(self, tokens: Sequence[str]) -> str:
        if self.delimiter_attachment == "discard":
            return self.detokenization_separator.join(tokens)
        return "".join(tokens)


@BaseTextAnalyzer.register("characters")
class CharacterTextAnalyzer(BaseTextAnalyzer):
    """Treat every Unicode code point as one surface token."""

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def detokenize(self, tokens: Sequence[str]) -> str:
        return "".join(tokens)
