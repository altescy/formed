import re
from typing import ClassVar, Optional, Pattern

from collatable.utils import debatched  # noqa: F401


class RegexTokenizer:
    _DEFAULT_TOKENIZER_PATTERN: ClassVar[Pattern] = re.compile(r"[^\s.,!?:;/]+(?:[-']\[^\s.,!?:;/]+)*|[.,!?:;/]")

    def __init__(self, pattern: Optional[Pattern] = None) -> None:
        self._token_pattern = pattern or self._DEFAULT_TOKENIZER_PATTERN

    def __call__(self, text: str) -> list[str]:
        return self._token_pattern.findall(text)
