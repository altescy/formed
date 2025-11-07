import base64
import dataclasses
import datetime
import enum
import importlib
import json
from collections import Counter
from contextlib import suppress
from typing import Any, Final, cast

import cloudpickle
import colt
from pydantic import BaseModel

from formed.common.base58 import b58encode
from formed.common.hashutils import hash_object_bytes
from formed.common.typeutils import is_namedtuple
from formed.types import IJsonSerializable, JsonValue

_PYTHON_DATA_TYPE_KEY: Final = "__python_type__"
_PYTHON_DATA_VALUE_KEY: Final = "__python_value__"
_PYTHON_DATA_CONTAINER_KEY: Final = "__python_container__"


def object_fingerprint(obj: Any) -> str:
    with suppress(TypeError, ValueError):
        # This is a workaround for fingerprint consistency.
        obj = json.loads(json.dumps(obj, cls=WorkflowJSONEncoder, sort_keys=True))
    return b58encode(hash_object_bytes(obj)).decode()


def as_jsonvalue(value: Any) -> JsonValue:
    return cast(JsonValue, json.loads(json.dumps(value, cls=WorkflowJSONEncoder)))


class _JSONDataType(str, enum.Enum):
    CLASS = "class"
    CONTAINER = "container"
    COUNTER = "counter"
    DATETIME = "datetime"
    PICKLE = "pickle"


class WorkflowJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        from .colt import WorkflowRef
        from .graph import WorkflowGraph
        from .step import WorkflowStepInfo

        if isinstance(o, (WorkflowGraph, WorkflowStepInfo)):
            return o.to_dict()
        if isinstance(o, datetime.datetime):
            return o.isoformat()
        if is_namedtuple(o):
            return o._asdict()
        if isinstance(o, tuple):
            return list(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, WorkflowRef):
            return o.config
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, BaseModel):
            return json.loads(o.model_dump_json())
        return super().default(o)
