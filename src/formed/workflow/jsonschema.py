from typing import Any, Final, Mapping, TypedDict

from formed.common.jsonschema import generate_json_schema
from formed.workflow import WorkflowStep

from .constants import WORKFLOW_REFKEY, WORKFLOW_REFTYPE

_REF_SCHEMA: Final[Mapping[str, Any]] = {
    "type": "object",
    "properties": {
        "type": {"const": WORKFLOW_REFTYPE},
        WORKFLOW_REFKEY: {"type": "string"},
    },
}
_REF_SCHEMA_REF: Final[Mapping[str, Any]] = {"$ref": "#/$defs/__ref__"}


def _ref_callback(path: str | None, schema: dict[str, Any]) -> dict[str, Any]:
    if path and path.startswith("steps.") and "const" not in schema and _REF_SCHEMA_REF not in schema.get("anyOf", []):
        return {"anyOf": [schema, _REF_SCHEMA_REF]}
    return schema


def generate_workflow_schema(
    title: str = "Formed Workflow Graph",
) -> dict[str, Any]:
    WorkflowGraphSchema = TypedDict("WorkflowGraphSchema", {"steps": dict[str, WorkflowStep]})
    return generate_json_schema(
        WorkflowGraphSchema,
        definitions={"__ref__": _REF_SCHEMA},
        callback=_ref_callback,
        title=title,
    )
