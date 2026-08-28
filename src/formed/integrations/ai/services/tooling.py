from collections.abc import Callable
from typing import Final, TypeVar

from colt.builder import ColtBuilder
from colt.error import ConfigurationError
from colt.jsonschema import JsonSchemaGenerator

from formed.types import JsonValue

from ..entities.tools import ToolDefinition, ToolSpec

_R = TypeVar("_R")
_COLT_BUILDER: Final = ColtBuilder(strict=True)
_JSON_SCHEMA_GENERATOR: Final = JsonSchemaGenerator(strict=True)


class InvalidToolArgumentsError(Exception):
    """Raised when the arguments provided to a tool are invalid."""


def build_tool_definition(
    func_or_class: Callable[..., _R] | type,
    *,
    name: str | None = None,
    description: str | None = None,
) -> ToolDefinition[_R]:
    name = name or getattr(func_or_class, "__name__", None)
    description = description or getattr(func_or_class, "__doc__")

    if name is None:
        raise ValueError("Function or class must have a name")

    parameters = _JSON_SCHEMA_GENERATOR(func_or_class)

    spec = ToolSpec(
        name=name,
        description=description or "",
        parameters=parameters,
    )

    return ToolDefinition(spec=spec, func=func_or_class)


def execute_tool_with_params_json(tool_def: ToolDefinition[_R], args: JsonValue) -> _R:
    try:
        return _COLT_BUILDER(args, tool_def.func)
    except ConfigurationError as e:
        raise InvalidToolArgumentsError(f"Invalid arguments for tool '{tool_def.name}': {e}") from e
