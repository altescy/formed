from __future__ import annotations

import dataclasses

import pytest

from formed.integrations.ai.providers.base.tools import Toolset


@dataclasses.dataclass(frozen=True)
class _Payload:
    value: int


def _sample_tool(a: int, payload: _Payload) -> str:
    """Sample tool used for schema generation tests."""
    return f"{a + payload.value}"


def test_toolset_generates_schema_from_callable_annotations() -> None:
    toolset = Toolset.from_tools((_sample_tool,))

    definition = toolset.definitions[0]
    assert definition.name == "_sample_tool"
    assert definition.strict is True
    assert definition.parameters["type"] == "object"
    assert "a" in definition.parameters["properties"]
    assert "payload" in definition.parameters["properties"]


def test_toolset_supports_named_mapping_inputs() -> None:
    toolset = Toolset.from_tools({"sum_payload": _sample_tool})

    assert toolset.definitions[0].name == "sum_payload"
    assert toolset.get("sum_payload") is _sample_tool


def test_toolset_accepts_sequence_of_callables() -> None:
    toolset = Toolset.from_tools((_sample_tool,))

    assert toolset.definitions[0].name == "_sample_tool"
    assert toolset.get("_sample_tool") is _sample_tool


def test_toolset_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError):
        Toolset.from_tools((_sample_tool, _sample_tool))


def test_toolset_supports_custom_schema_generator() -> None:
    class _SchemaGenerator:
        def __call__(
            self,
            target: object,
            *,
            title: str | None = None,
            description: str | None = None,
            definitions: dict[str, object] | None = None,
        ) -> dict[str, object]:
            del target, title, description, definitions
            return {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            }

    toolset = Toolset.from_tools((_sample_tool,), schema_generator=_SchemaGenerator())
    assert toolset.definitions[0].parameters["properties"] == {"x": {"type": "integer"}}
