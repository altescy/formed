from typing import Any, Final, Mapping, TypedDict

from colt.jsonschema import JsonSchemaContext, JsonSchemaGenerator

from formed.constants import COLT_ARGSKEY, COLT_TYPEKEY
from formed.workflow import WorkflowStep

from .constants import WORKFLOW_REFKEY, WORKFLOW_REFTYPE

_REF_SCHEMA: Final[Mapping[str, Any]] = {
    "type": "object",
    "properties": {
        "type": {"const": WORKFLOW_REFTYPE},
        WORKFLOW_REFKEY: {"type": "string"},
    },
    "required": ["type", WORKFLOW_REFKEY],
    "additionalProperties": False,
}
_REF_SCHEMA_REF: Final[Mapping[str, Any]] = {"$ref": "#/$defs/__ref__"}


def _is_ref_schema(schema: dict[str, Any]) -> bool:
    return schema == _REF_SCHEMA_REF


def _is_importable_schema(schema: dict[str, Any]) -> bool:
    return (
        schema.get("type") == "object"
        and schema.get("properties") == {"type": {"type": "string"}}
        and schema.get("additionalProperties") is True
    )


def _remove_importable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if "anyOf" not in schema:
        return schema
    subschemas = [s for s in schema["anyOf"] if not _is_importable_schema(s)]
    if not subschemas:
        return schema
    if len(subschemas) == 1:
        return subschemas[0]
    return {"anyOf": subschemas}


def _remove_ref_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if "anyOf" not in schema:
        return schema
    subschemas = [s for s in schema["anyOf"] if not _is_ref_schema(s)]
    if not subschemas:
        return schema
    if len(subschemas) == 1:
        return subschemas[0]
    return {"anyOf": subschemas}


def _ref_callback(schema: dict[str, Any], context: JsonSchemaContext) -> dict[str, Any]:
    if context.path <= ("steps", "<key>"):
        return _remove_importable_schema(schema)
    return {"anyOf": [schema, _REF_SCHEMA_REF]}


def generate_workflow_schema(
    title: str = "formed Workflow Graph",
) -> dict[str, Any]:
    WorkflowGraphSchema = TypedDict("WorkflowGraphSchema", {"steps": dict[str, WorkflowStep]})
    definitions = {"__ref__": _REF_SCHEMA}
    generator = JsonSchemaGenerator(
        callback=_ref_callback,
        typekey=COLT_TYPEKEY,
        argskey=COLT_ARGSKEY,
    )
    schema = generator(
        WorkflowGraphSchema,
        definitions=definitions,
        title=title,
    )
    if defs := schema.get("$defs", None):
        for key, defschema in defs.items():
            defschema = _remove_ref_schema(defschema)
            def_schema = _remove_importable_schema(defschema)
            defs[key] = def_schema
    return schema
