import dataclasses
import inspect
import string
import typing
from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence, Set
from enum import Enum
from types import NoneType, UnionType
from typing import Any, Dict, Final, List, Literal, Optional, Union

from colt import Lazy, Registrable
from typing_extensions import TypeAlias

from formed.common.typeutils import is_namedtuple

_SAFE_CHARS: Final = frozenset(string.ascii_letters + string.digits + "_")

JsonType: TypeAlias = Literal["string", "number", "integer", "boolean", "object", "array", "null"]


def _default(python_type: Any) -> dict[str, Any]:
    del python_type
    return {
        "type": "object",
        "additionalProperties": True,
        "description": "default schema for unsupported type",
    }


def generate_json_schema(
    cls: Any,
    *,
    root: bool = True,
    definitions: Optional[dict[str, Any]] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    meta_properties: Optional[Mapping[str, Any]] = None,
    default: Union[Mapping[str, Any], Callable[[Any], dict[str, Any]]] = _default,
    callback: Optional[Callable[[Optional[str], dict[str, Any]], dict[str, Any]]] = None,
    path: Optional[str] = None,
) -> dict[str, Any]:
    if cls is Any:
        return {}

    if definitions is None:
        definitions = {}

    schema: Dict[str, Any] | None = None
    ref_name: str | None = None

    if isinstance(cls, type):
        if isinstance(cls, type) and issubclass(cls, Enum):
            schema = {"enum": [member.value for member in cls]}
        elif cls in (int, float, str, bool, NoneType):
            schema = {"type": _get_json_type(cls)}
        elif cls in (list, tuple, set, List, Sequence, MutableSequence, Set):
            schema = {"type": "array"}
        elif cls in (dict, Dict, Mapping, MutableMapping):
            schema = {"type": "object"}
        elif issubclass(cls, Registrable) and (registry := Registrable._registry.get(cls)):
            schema = {
                "anyOf": [
                    generate_json_schema(
                        subclass,
                        root=False,
                        definitions=definitions,
                        meta_properties={"type": {"const": name}},
                        default=default,
                        callback=callback,
                        path=_concat_path(path, name),
                    )
                    for name, (subclass, _) in registry.items()
                ]
            }
        else:
            ref_name = _get_ref_name(cls)
            if not root and ref_name in definitions:
                return {"$ref": f"#/$defs/{ref_name}"}
            if dataclasses.is_dataclass(cls):
                fields = [field for field in dataclasses.fields(cls) if field.init]
                schema = {
                    "type": "object",
                    "properties": {
                        field.name: generate_json_schema(
                            field.type,
                            root=False,
                            definitions=definitions,
                            default=default,
                            callback=callback,
                            path=_concat_path(path, field.name),
                        )
                        for field in fields
                    },
                    "required": [
                        field.name
                        for field in fields
                        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
                    ],
                }
            elif is_namedtuple(cls):
                annotations = typing.get_type_hints(cls)
                schema = {
                    "type": "object",
                    "properties": {
                        name: generate_json_schema(
                            annotation,
                            root=False,
                            definitions=definitions,
                            default=default,
                            callback=callback,
                            path=_concat_path(path, name),
                        )
                        for name, annotation in typing.get_type_hints(cls).items()
                    },
                    "required": [name for name in annotations.keys() if name not in cls._field_defaults],
                }
            elif issubclass(cls, dict) and (annotations := typing.get_type_hints(cls)):
                schema = {
                    "type": "object",
                    "properties": {
                        name: generate_json_schema(
                            annotation,
                            root=False,
                            definitions=definitions,
                            default=default,
                            callback=callback,
                            path=_concat_path(path, name),
                        )
                        for name, annotation in annotations.items()
                    },
                    "required": [name for name in annotations.keys() if not hasattr(cls, name)],
                }
            elif hasattr(cls, "__init__"):
                schema = generate_json_schema(
                    cls.__init__,
                    root=False,
                    definitions=definitions,
                    default=default,
                    callback=callback,
                    path=path,
                )

            if schema is not None:
                title = title or cls.__qualname__
    elif origin := typing.get_origin(cls):
        args = typing.get_args(cls)
        if origin in (Union, UnionType):  # for Optional and UnionType
            types = [
                generate_json_schema(
                    t,
                    root=False,
                    definitions=definitions,
                    default=default,
                    callback=callback,
                    path=path,
                )
                for t in cls.__args__
                if t is not NoneType
            ]  # exclude None
            schema = (
                {"anyOf": types} if len(types) > 1 else types[0]
            )  # if only one type excluding None, no need to use array
        elif origin in (list, List, Sequence, MutableSequence):  # for List
            schema = {"type": "array"}
            if args:
                schema["items"] = generate_json_schema(
                    args[0],
                    root=False,
                    definitions=definitions,
                    default=default,
                    callback=callback,
                    path=path,
                )
        elif origin in (dict, Dict, Mapping, MutableMapping):  # for Dict
            schema = {"type": "object"}
            if len(args) == 2 and args[0] is str:
                schema["additionalProperties"] = generate_json_schema(
                    args[1],
                    root=False,
                    definitions=definitions,
                    default=default,
                    callback=callback,
                    path=path,
                )
        elif origin is Literal:  # for Literal
            schema = {"enum": list(args)}
        elif origin is Lazy:
            schema = generate_json_schema(
                args[0] if args else Any,
                root=False,
                definitions=definitions,
                default=default,
                callback=callback,
                path=path,
            )
        else:
            schema = generate_json_schema(
                origin,
                root=False,
                definitions=definitions,
                default=default,
                callback=callback,
                path=path,
            )
    elif callable(cls):
        sig = inspect.signature(cls)
        params = [
            (name, param)
            for pos, (name, param) in enumerate(sig.parameters.items())
            if not (pos == 0 and name in ("self", "cls"))
            and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        schema = {
            "type": "object",
            "properties": {
                name: generate_json_schema(
                    param.annotation if param.annotation is not inspect.Parameter.empty else Any,
                    root=False,
                    definitions=definitions,
                    default=default,
                    callback=callback,
                    path=_concat_path(path, name),
                )
                for name, param in params
            },
            "required": [name for name, param in params if param.default is inspect.Parameter.empty],
        }

    if schema is None:
        schema = default(cls) if callable(default) else dict(default)

    if meta_properties:
        if schema.get("type") == "object":
            schema.setdefault("properties", {}).update(meta_properties)
            schema.setdefault("required", []).extend(meta_properties.keys())

    if callback:
        schema = callback(path, schema)

    if title:
        schema["title"] = title
    if description:
        schema["description"] = description

    # Register class schema as reference
    if not root and schema is not None and ref_name is not None:
        definitions[ref_name] = schema
        schema = {"$ref": f"#/$defs/{ref_name}"}

    if root:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": definitions,
            **schema,
        }

    return schema


def _get_ref_name(python_type: type) -> str:
    return f"{_safe_name(python_type.__module__)}__{_safe_name(python_type.__qualname__)}"


def _safe_name(name: str) -> str:
    return "".join(c if c in _SAFE_CHARS else "__" for c in name)


def _concat_path(path: Optional[str], segment: str) -> str:
    if path:
        return f"{path}.{segment}"
    return segment


def _get_json_type(python_type: type) -> JsonType:
    if python_type is int:
        return "integer"
    elif python_type is float:
        return "number"
    elif python_type is str:
        return "string"
    elif python_type is bool:
        return "boolean"
    elif python_type is NoneType:
        return "null"
    elif python_type is list:
        return "array"
    return "object"
