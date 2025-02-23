from collections.abc import Mapping, Sequence
from typing import Any, Union, overload

import jax
import numpy


@overload
def numpy_to_jax(x: Union[int, float]) -> jax.numpy.ndarray: ...


@overload
def numpy_to_jax(x: numpy.ndarray) -> jax.numpy.ndarray: ...


@overload
def numpy_to_jax(x: Sequence[Any]) -> list[Any]: ...


@overload
def numpy_to_jax(x: Mapping[Any, Any]) -> dict[Any, Any]: ...


def numpy_to_jax(x: Union[numpy.ndarray, Sequence[Any], Mapping[Any, Any], Any]) -> Any:
    if isinstance(x, (int, float)):
        return jax.numpy.array(x)
    if isinstance(x, numpy.ndarray):
        return jax.numpy.array(x)
    if isinstance(x, Sequence):
        return [numpy_to_jax(item) for item in x]
    if isinstance(x, Mapping):
        return {key: numpy_to_jax(value) for key, value in x.items()}
    return x
