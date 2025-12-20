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
import numpy

from formed.common.base58 import b58encode
from formed.common.hashutils import hash_object_bytes
from formed.common.typeutils import is_namedtuple
from formed.types import IJsonCompatible, IJsonDeserializable, IJsonSerializable, IPydanticModel, JsonValue

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


def from_jsonvalue(value: JsonValue) -> Any:
    return WorkflowJSONDecoder()._reconstruct(value)


class _JSONDataType(str, enum.Enum):
    CLASS = "class"
    CONTAINER = "container"
    COUNTER = "counter"
    DATETIME = "datetime"
    PICKLE = "pickle"
    NDARRAY = "ndarray"


class WorkflowJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, IJsonCompatible):
            return {
                _PYTHON_DATA_TYPE_KEY: _JSONDataType.PICKLE,
                _PYTHON_DATA_VALUE_KEY: o.json(),
                _PYTHON_DATA_CONTAINER_KEY: f"{o.__class__.__module__}.{o.__class__.__qualname__}",
            }
        if isinstance(o, IJsonSerializable):
            return o.json()
        if isinstance(o, datetime.datetime):
            return {_PYTHON_DATA_TYPE_KEY: _JSONDataType.DATETIME, _PYTHON_DATA_VALUE_KEY: o.isoformat()}
        if is_namedtuple(o):
            return {
                _PYTHON_DATA_TYPE_KEY: _JSONDataType.CONTAINER,
                _PYTHON_DATA_VALUE_KEY: o._asdict(),
                _PYTHON_DATA_CONTAINER_KEY: f"{o.__class__.__module__}.{o.__class__.__qualname__}",
            }
        if isinstance(o, tuple):
            return {
                _PYTHON_DATA_TYPE_KEY: _JSONDataType.CONTAINER,
                _PYTHON_DATA_VALUE_KEY: [self.default(i) for i in o],
                _PYTHON_DATA_CONTAINER_KEY: f"{o.__class__.__module__}.{o.__class__.__qualname__}",
            }
        if isinstance(o, (set, frozenset)):
            return {
                _PYTHON_DATA_TYPE_KEY: _JSONDataType.CONTAINER,
                _PYTHON_DATA_VALUE_KEY: sorted((self.default(i) for i in o), key=lambda x: (hash(x), str(x))),
                _PYTHON_DATA_CONTAINER_KEY: f"{o.__class__.__module__}.{o.__class__.__qualname__}",
            }
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return {
                _PYTHON_DATA_TYPE_KEY: _JSONDataType.CONTAINER,
                _PYTHON_DATA_VALUE_KEY: {
                    field.name: self.default(getattr(o, field.name))
                    for field in dataclasses.fields(o)
                    if hasattr(o, field.name)
                },
                _PYTHON_DATA_CONTAINER_KEY: f"{o.__class__.__module__}.{o.__class__.__qualname__}",
            }
        if isinstance(o, IPydanticModel):
            return {
                _PYTHON_DATA_TYPE_KEY: _JSONDataType.CONTAINER,
                _PYTHON_DATA_VALUE_KEY: o.model_dump(mode="json"),
                _PYTHON_DATA_CONTAINER_KEY: f"{o.__class__.__module__}.{o.__class__.__qualname__}",
            }
        if isinstance(o, type):
            return {
                _PYTHON_DATA_TYPE_KEY: _JSONDataType.CLASS,
                _PYTHON_DATA_VALUE_KEY: f"{o.__module__}.{o.__qualname__}",
            }
        if isinstance(o, Counter):
            return {
                _PYTHON_DATA_TYPE_KEY: _JSONDataType.COUNTER,
                _PYTHON_DATA_VALUE_KEY: dict(o),
            }
        if isinstance(o, numpy.ndarray):
            return {
                _PYTHON_DATA_TYPE_KEY: _JSONDataType.NDARRAY,
                _PYTHON_DATA_VALUE_KEY: {
                    "dtype": str(o.dtype),
                    "shape": o.shape,
                    "data": base64.b85encode(o.tobytes()).decode(),
                },
            }
        if isinstance(o, list):
            return [self.default(i) for i in o]
        if isinstance(o, dict):
            return {k: self.default(v) for k, v in o.items()}
        if isinstance(o, (bool, int, float, str)) or o is None:
            return o
        return {
            _PYTHON_DATA_TYPE_KEY: _JSONDataType.PICKLE,
            _PYTHON_DATA_VALUE_KEY: base64.b85encode(cloudpickle.dumps(o)).decode(),
        }


class WorkflowJSONDecoder(json.JSONDecoder):
    def __init__(self) -> None:
        super().__init__(object_hook=self._reconstruct)

    @staticmethod
    def _reconstruct(o) -> Any:
        if isinstance(o, list):
            return [WorkflowJSONDecoder._reconstruct(i) for i in o]
        if not isinstance(o, dict):
            return o
        if _PYTHON_DATA_TYPE_KEY not in o:
            return {k: WorkflowJSONDecoder._reconstruct(v) for k, v in o.items()}
        data_type = _JSONDataType(o[_PYTHON_DATA_TYPE_KEY])
        if data_type == _JSONDataType.CLASS:
            module_name, class_name = o[_PYTHON_DATA_VALUE_KEY].rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        if data_type == _JSONDataType.CONTAINER:
            module_name, class_name = o[_PYTHON_DATA_CONTAINER_KEY].rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            value: dict = WorkflowJSONDecoder._reconstruct(o[_PYTHON_DATA_VALUE_KEY])
            if isinstance(cls, type) and dataclasses.is_dataclass(cls):
                init_fields = []
                non_init_fields = []
                for field in dataclasses.fields(cls):
                    if field.init:
                        init_fields.append(field)
                    else:
                        non_init_fields.append(field)
                output = colt.build(
                    {field.name: value[field.name] for field in init_fields if field.name in value}, cls
                )
                for field in non_init_fields:
                    if field.name in value:
                        setattr(output, field.name, colt.build(value[field.name], field.type))
                    elif field.default is not dataclasses.MISSING:
                        setattr(output, field.name, field.default)
                    elif field.default_factory is not dataclasses.MISSING:
                        setattr(output, field.name, field.default_factory())
                return output
            if isinstance(cls, type) and issubclass(cls, IJsonDeserializable):
                return cls.from_json(value)
            if isinstance(cls, type) and issubclass(cls, IPydanticModel):
                return cls.model_validate(value)
            return colt.build(value, cls)
        if data_type == _JSONDataType.COUNTER:
            return Counter(o[_PYTHON_DATA_VALUE_KEY])
        if data_type == _JSONDataType.DATETIME:
            return datetime.datetime.fromisoformat(o[_PYTHON_DATA_VALUE_KEY])
        if data_type == _JSONDataType.PICKLE:
            return cloudpickle.loads(base64.b85decode(o[_PYTHON_DATA_VALUE_KEY].encode()))
        if data_type == _JSONDataType.NDARRAY:
            array_info = o[_PYTHON_DATA_VALUE_KEY]
            data_bytes = base64.b85decode(array_info["data"].encode())
            return numpy.frombuffer(data_bytes, dtype=array_info["dtype"]).reshape(array_info["shape"])
        raise ValueError(f"Unknown data type: {data_type}")
