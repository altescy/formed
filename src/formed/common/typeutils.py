from typing import Any, TypeGuard

from formed.types import INamedTuple


def is_namedtuple(obj: Any) -> TypeGuard[INamedTuple]:
    if not isinstance(obj, type) and isinstance(obj, object):
        obj = type(obj)
    if not isinstance(obj, type):
        return False
    if not issubclass(obj, tuple):
        return False
    fields = getattr(obj, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(type(name) is str for name in fields)
