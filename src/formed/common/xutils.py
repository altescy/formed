from collections.abc import Mapping, Sequence
from typing import Any, Optional, TypeVar

_T = TypeVar("_T")
_NotSpecified = object()


def xgetattr(
    o: Any,
    /,
    name: str,
    default: Any = _NotSpecified,
) -> Any:
    try:
        child: Optional[str] = None
        if "." in name:
            name, child = name.split(".", 1)
        if name == "*":
            assert isinstance(o, Sequence) and child, "Wildcard field must be used with a child field"
            return [xgetattr(item, child) for item in o]
        if isinstance(o, Mapping):
            o = o[name]
        else:
            o = getattr(o, name)
        if child:
            o = xgetattr(o, child)
        return o
    except (KeyError, AttributeError):
        if default is _NotSpecified:
            raise
        return default
