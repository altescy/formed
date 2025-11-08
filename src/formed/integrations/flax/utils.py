import dataclasses
from functools import cache
from typing import Any

import jax
import numpy
from typing_extensions import TypeVar

from formed.common.jax import JAX_STATIC_FIELD
from formed.integrations.ml.transforms.base import DataModule

_DataModuleTypeT = TypeVar("_DataModuleTypeT", bound=type[DataModule[Any, Any, Any, Any]])


def ensure_jax_array(x: numpy.ndarray) -> jax.Array:
    if isinstance(x, jax.Array):
        return x
    return jax.numpy.asarray(x)


@cache
def register_datamodule(nodetype: _DataModuleTypeT) -> _DataModuleTypeT:
    drop_fields = [f.name for f in dataclasses.fields(nodetype) if not f.init]
    data_fields = [
        f.name
        for f in dataclasses.fields(nodetype)
        if not f.metadata.get(JAX_STATIC_FIELD, False) and f.name not in drop_fields
    ]
    meta_fields = [
        f.name
        for f in dataclasses.fields(nodetype)
        if f.metadata.get(JAX_STATIC_FIELD, False) and f.name not in drop_fields
    ]

    try:
        return jax.tree_util.register_dataclass(
            nodetype,
            data_fields=data_fields,
            meta_fields=meta_fields,
            drop_fields=drop_fields,
        )
    except ValueError as error:
        if str(error.args[0]).startswith("Duplicate custom dataclass"):
            return nodetype
        raise
