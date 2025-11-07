from collections.abc import Mapping, Sequence
from typing import Any, Optional

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
        if name not in _get_available_attributes(o):
            if default is _NotSpecified:
                raise AttributeError(f"{type(o).__name__!r} has no attribute {name!r}")
            return default
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


def _get_available_attributes(o: Any) -> set[str]:
    attrs = set()

    if isinstance(o, Mapping):
        attrs |= set(o.keys())
    else:
        attrs |= {name for name in dir(o) if not name.startswith("_")}

    slots = getattr(o, "__slots__", ())
    annotations = getattr(o, "__annotations__", {})
    if annotations:
        attrs |= set(annotations.keys())
    elif slots:
        attrs = attrs & set(slots)

    return attrs
