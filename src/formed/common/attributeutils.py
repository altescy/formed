"""Extended attribute access utilities.

This module provides `xgetattr`, an enhanced version of Python's built-in `getattr`
that supports nested attribute access, dictionary key access, and wildcard patterns.

Key Features:
    - Dot-separated nested attribute access
    - Unified dict/object attribute access
    - Wildcard pattern for mapping over sequences
    - Default value support

Examples:
    >>> from formed.common.attributeutils import xgetattr
    >>>
    >>> # Nested attribute access
    >>> obj = type('Obj', (), {'user': type('User', (), {'name': 'Alice'})()})()
    >>> xgetattr(obj, "user.name")  # "Alice"
    >>>
    >>> # Dictionary access
    >>> data = {"config": {"model": {"name": "bert"}}}
    >>> xgetattr(data, "config.model.name")  # "bert"
    >>>
    >>> # Wildcard mapping
    >>> users = [{"name": "Alice"}, {"name": "Bob"}]
    >>> xgetattr(users, "*.name")  # ["Alice", "Bob"]

"""

from collections.abc import Mapping, Sequence
from typing import Any

_NotSpecified = object()


def xgetattr(
    o: Any,
    /,
    name: str,
    default: Any = _NotSpecified,
) -> Any:
    """Extended attribute getter with nested access and wildcard support.

    This function extends Python's `getattr()` to support:
    - Nested attributes via dot notation (e.g., "user.profile.name")
    - Dictionary key access alongside object attributes
    - Wildcard "*" for mapping over sequences

    Args:
        o: The object to get the attribute from.
        name: Attribute name or dot-separated path. Use "*" to map over sequences.
        default: Default value if attribute is not found. If not specified,
            raises AttributeError on missing attributes.

    Returns:
        The requested attribute value. For wildcards, returns a list.

    Raises:
        AttributeError: If attribute not found and no default specified.

    Examples:
        >>> # Nested object access
        >>> class User:
        ...     def __init__(self, name):
        ...         self.name = name
        >>> class Config:
        ...     def __init__(self, user):
        ...         self.user = user
        >>> config = Config(User("Alice"))
        >>> xgetattr(config, "user.name")
        'Alice'
        >>>
        >>> # Dict access
        >>> data = {"step": {"output": {"value": 42}}}
        >>> xgetattr(data, "step.output.value")
        42
        >>>
        >>> # Wildcard mapping
        >>> items = [{"id": 1}, {"id": 2}, {"id": 3}]
        >>> xgetattr(items, "*.id")
        [1, 2, 3]
        >>>
        >>> # Default value
        >>> xgetattr(config, "missing.attr", default=None)
        None

    Note:
        - Wildcards require the object to be a Sequence with a child field
        - Dict keys and object attributes are accessed uniformly
        - Annotation-based attributes (__annotations__) are recognized

    """
    try:
        child: str | None = None
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
