from typing import TypeVar

from sentence_transformers import SentenceTransformer

SentenceTransformerT = TypeVar("SentenceTransformerT", bound=SentenceTransformer)
