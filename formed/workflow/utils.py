import dataclasses
import datetime
import json
from contextlib import suppress
from typing import Any, cast

from pydantic import BaseModel

from formed.common.base58 import b58encode
from formed.common.hashutils import hash_object_bytes
from formed.common.typeutils import is_namedtuple
from formed.types import JsonValue


def object_fingerprint(obj: Any) -> str:
    with suppress(TypeError, ValueError):
        # This is a workaround for fingerprint consistency.
        obj = json.loads(json.dumps(obj, cls=WorkflowJSONEncoder, sort_keys=True))
    return b58encode(hash_object_bytes(obj)).decode()


def as_jsonvalue(value: Any) -> JsonValue:
    return cast(JsonValue, json.loads(json.dumps(value, cls=WorkflowJSONEncoder)))


class WorkflowJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        from .colt import WorkflowRef
        from .graph import WorkflowGraph
        from .step import WorkflowStepInfo

        if isinstance(obj, (WorkflowGraph, WorkflowStepInfo)):
            return obj.to_dict()
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if is_namedtuple(obj):
            return obj._asdict()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        if isinstance(obj, WorkflowRef):
            return obj.config
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, BaseModel):
            return json.loads(obj.model_dump_json())
        return super().default(obj)
