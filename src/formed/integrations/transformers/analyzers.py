"""Text analyzers using pretrained transformers tokenizers.

This module provides text analysis tools that leverage pretrained tokenizers
from the Hugging Face transformers library to tokenize text into surface forms.

Available Classes:
    - PretrainedAnalyzer: Analyzer using pretrained transformer tokenizers

Example:
    >>> from formed.integrations.transformers.analyzers import PretrainedAnalyzer
    >>>
    >>> # Initialize with model name
    >>> analyzer = PretrainedAnalyzer("bert-base-uncased")
    >>> result = analyzer("Hello world!")
    >>> print(result.surfaces)
    ['hello', 'world', '!']

"""

import dataclasses
from collections.abc import Sequence
from functools import cached_property
from os import PathLike

from transformers import PreTrainedTokenizerBase

from formed.integrations.ml.types import AnalyzedText


@dataclasses.dataclass
class PretrainedTransformerAnalyzer:
    """Text analyzer using pretrained transformer tokenizers.

    This analyzer uses tokenizers from the Hugging Face transformers library
    to split text into tokens (surface forms). It provides a simple interface
    for text tokenization that's compatible with the formed ML pipeline.

    Args:
        tokenizer: Either a tokenizer name/path string or a PreTrainedTokenizerBase instance.
            If a string, the tokenizer will be loaded using AutoTokenizer.

    Example:
        >>> # Initialize with model name
        >>> analyzer = PretrainedAnalyzer("bert-base-uncased")
        >>> result = analyzer("Hello, world!")
        >>> print(result.surfaces)
        ['hello', ',', 'world', '!']
        >>>
        >>> # Initialize with tokenizer instance
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        >>> analyzer = PretrainedAnalyzer(tokenizer)
        >>> result = analyzer("Machine learning is great!")
        >>> print(result.surfaces)
        ['Machine', 'Ġlearning', 'Ġis', 'Ġgreat', '!']

    Note:
        Tokenizers are cached using LRU cache by the load_pretrained_tokenizer utility.
        The returned AnalyzedText only contains surface forms; other fields like
        postags are None.

    """

    tokenizer: str | PathLike | PreTrainedTokenizerBase

    @cached_property
    def _tokenizer(self) -> PreTrainedTokenizerBase:
        if isinstance(self.tokenizer, (str, PathLike)):
            from .utils import load_pretrained_tokenizer

            return load_pretrained_tokenizer(self.tokenizer)
        return self.tokenizer

    def __call__(self, text: str | Sequence[str] | AnalyzedText) -> AnalyzedText:
        if isinstance(text, AnalyzedText):
            return text
        if isinstance(text, str):
            surfaces = self._tokenizer.tokenize(text)
        elif isinstance(text, Sequence):
            surfaces = text
        return AnalyzedText(surfaces=surfaces)
