import re
from typing import Final

_PUNKT_SPLIT_PATTERN: Final = re.compile(r"[,.!?@/\-\'\"\s\(\)\[\]{}]+")


def punkt_tokenize(text: str) -> list[str]:
    """A simple tokenizer that splits text into tokens based on punctuation and whitespace.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list[str]: A list of tokens extracted from the input text.

    """
    return list(filter(bool, _PUNKT_SPLIT_PATTERN.split(text)))
