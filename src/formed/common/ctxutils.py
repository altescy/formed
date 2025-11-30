from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

from formed.types import SupportsClosing

T = TypeVar("T")


@contextmanager
def closing(thing: T) -> Iterator[T]:
    try:
        yield thing
    finally:
        if isinstance(thing, SupportsClosing):
            thing.close()
