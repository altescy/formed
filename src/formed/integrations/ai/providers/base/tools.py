from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol, TypeAlias

from colt import JsonSchemaGenerator

ToolImplementation: TypeAlias = Callable[..., object | Awaitable[object]]


class ToolSchemaGenerator(Protocol):
    """Protocol for generating JSON Schema from a callable."""

    def __call__(
        self,
        target: Any,
        *,
        title: str | None = None,
        description: str | None = None,
        definitions: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class ToolArgsBuilder(Protocol):
    """Protocol for building typed callable arguments from JSON-like values."""

    def __call__(
        self,
        config: Any,
        cls: Callable[..., object],
    ) -> Any: ...


@dataclasses.dataclass(frozen=True)
class Toolset:
    """Immutable container for tool definitions and implementations."""

    definitions: tuple[ToolDefinition, ...]
    _implementations: Mapping[str, ToolImplementation] = dataclasses.field(repr=False)

    @classmethod
    def empty(cls) -> Toolset:
        return cls(definitions=(), _implementations={})

    @classmethod
    def from_tools(
        cls,
        tools: Mapping[str, ToolImplementation] | Sequence[ToolLike],
        *,
        definition_strict: bool | None = True,
        schema_generator: ToolSchemaGenerator | None = None,
    ) -> Toolset:
        definitions, implementations = _normalize_tools(
            tools,
            definition_strict=definition_strict,
            schema_generator=schema_generator,
        )
        return cls(definitions=definitions, _implementations=implementations)

    def get(self, name: str) -> ToolImplementation | None:
        return self._implementations.get(name)


@dataclasses.dataclass(frozen=True)
class ToolDefinition:
    """Definition of a callable tool exposed to the model.

    Attributes:
        name: Tool name as the model will call it.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema object describing the arguments.
        strict: If ``True``, request strict schema adherence (e.g. OpenAI
            structured outputs). ``None`` means provider default.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool | None = None


@dataclasses.dataclass(frozen=True)
class BoundTool:
    """Pair of tool definition and executable implementation.

    Attributes:
        definition: Tool metadata exposed to the model.
        implementation: Python callable invoked by :class:`DefaultHandler`.
    """

    definition: ToolDefinition
    implementation: ToolImplementation


ToolLike: TypeAlias = ToolImplementation | BoundTool


def _bind_tool(
    implementation: ToolImplementation,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool | None = True,
    schema_generator: ToolSchemaGenerator | None = None,
) -> BoundTool:
    """Create a :class:`BoundTool` from a typed Python callable.

    ``JsonSchemaGenerator`` is used to derive the tool parameter schema from
    the callable signature and type annotations, allowing complex nested types
    (dataclasses, TypedDict, etc.) to be represented in JSON Schema.
    """

    tool_name = name or implementation.__name__
    tool_description = description or inspect.getdoc(implementation) or tool_name
    # Use non-strict schema generation by default so postponed annotations
    # (`from __future__ import annotations`) remain supported for nested types.
    generator = schema_generator or JsonSchemaGenerator(strict=False)
    schema = generator(implementation, title=tool_name, description=tool_description)

    parameters = {key: value for key, value in schema.items() if key not in {"$schema", "title", "description"}}
    if "type" not in parameters:
        parameters["type"] = "object"

    return BoundTool(
        definition=ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            strict=strict,
        ),
        implementation=implementation,
    )


def _split_bound_tools(tools: Sequence[BoundTool]) -> tuple[tuple[ToolDefinition, ...], dict[str, ToolImplementation]]:
    """Split bound tools into query-facing definitions and runtime map."""

    definitions: list[ToolDefinition] = []
    implementations: dict[str, ToolImplementation] = {}
    for tool in tools:
        definitions.append(tool.definition)
        implementations[tool.definition.name] = tool.implementation
    return tuple(definitions), implementations


def _normalize_tools(
    tools: Mapping[str, ToolImplementation] | Sequence[ToolLike],
    *,
    definition_strict: bool | None = True,
    schema_generator: ToolSchemaGenerator | None = None,
) -> tuple[tuple[ToolDefinition, ...], dict[str, ToolImplementation]]:
    """Normalize user tool input into definitions and runtime map.

    ``tools`` accepts either a mapping ``name -> callable`` or a sequence of
    callables / :class:`BoundTool` objects. Callable inputs are converted with
    :func:`_bind_tool`.
    """

    bound_tools: list[BoundTool] = []

    if isinstance(tools, Mapping):
        for name, implementation in tools.items():
            bound_tools.append(
                _bind_tool(
                    implementation,
                    name=name,
                    strict=definition_strict,
                    schema_generator=schema_generator,
                )
            )
    else:
        for tool in tools:
            if isinstance(tool, BoundTool):
                bound_tools.append(tool)
            else:
                bound_tools.append(_bind_tool(tool, strict=definition_strict, schema_generator=schema_generator))

    definitions, implementations = _split_bound_tools(bound_tools)
    if len(definitions) != len(implementations):
        raise ValueError("Duplicate tool names are not allowed.")
    return definitions, implementations
