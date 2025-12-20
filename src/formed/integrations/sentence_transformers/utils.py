from contextlib import suppress
from functools import lru_cache
from os import PathLike
from typing import Any

import minato
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=8)
def load_sentence_transformer(
    model_name_or_path: str | PathLike,
    **kwargs: Any,
) -> SentenceTransformer:
    with suppress(FileNotFoundError):
        model_name_or_path = minato.cached_path(model_name_or_path)
    return SentenceTransformer(str(model_name_or_path), **kwargs)
