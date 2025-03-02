from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, TypeVar, Union, cast, overload

import jax
import numpy

MappingT = TypeVar("MappingT", bound=Mapping)


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


def masked_pool(
    embeddings: jax.Array,
    mask: Optional[jax.Array] = None,
    pooling: Literal["mean", "max", "min", "sum", "hier", "first", "last"] = "mean",
    normalize: bool = False,
    window_size: Optional[int] = None,
) -> jax.Array:
    """
    Pool embeddings with a mask.

    Args:
        embeddings: Embeddings to pool of shape (batch_size, sequence_length, embedding_size).
        mask: Mask of shape (batch_size, sequence_length).
        pooling: Pooling method. Defaults to `"mean"`.
        normalize: Whether to normalize the embeddings before pooling. Defaults to `False`.
        window_size: Window size for hierarchical pooling. Defaults to `None`.
    """

    batch_size, sequence_length, embedding_size = embeddings.shape

    if normalize:
        embeddings = embeddings / (jax.numpy.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-13)

    if mask is None:
        mask = jax.numpy.ones((batch_size, sequence_length), dtype=bool)

    if pooling == "mean":
        return embeddings.sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1e-13)

    if pooling == "max":
        embeddings[~mask] = float("-inf")
        return embeddings.max(axis=1)

    if pooling == "min":
        embeddings[~mask] = float("inf")
        return embeddings.min(axis=1)

    if pooling == "sum":
        return embeddings.sum(axis=1)

    if pooling == "first":
        return embeddings[:, 0, :]

    if pooling == "last":
        batch_indices = jax.numpy.arange(batch_size)
        last_positions = mask.cumsum(axis=1).argmax(axis=1)
        return embeddings[batch_indices, last_positions, :]

    if pooling == "hier":

        def _hierarchical_pooling(vectors: jax.Array, mask: jax.Array) -> jax.Array:
            assert window_size is not None
            vectors = vectors[mask]
            if len(vectors) < window_size:
                return vectors.mean(0)
            output: jax.Array = -jax.numpy.inf * jax.numpy.ones(embedding_size)
            for offset in range(len(vectors) - window_size + 1):
                window = vectors[offset : offset + window_size]
                output = jax.numpy.maximum(output, window.mean(0))
            return output

        output: jax.Array = jax.numpy.array([_hierarchical_pooling(x, m) for x, m in zip(embeddings, mask)])
        return output

    raise ValueError(
        f"pooling must be one of 'mean', 'max', 'min', 'sum', 'hier', 'first', or 'last', but got {pooling}"
    )


@overload
def sequence_distribute(
    inputs: jax.Array,
) -> tuple[jax.Array, tuple[int, int]]: ...


@overload
def sequence_distribute(
    inputs: MappingT,
    ignore: Sequence[str] = ...,
) -> tuple[MappingT, tuple[int, int]]: ...


def sequence_distribute(
    inputs: Union[jax.Array, MappingT],
    ignore: Sequence[str] = (),
) -> tuple[Union[jax.Array, MappingT], tuple[int, int]]:
    if isinstance(inputs, jax.Array):
        if inputs.ndim < 2:
            return inputs, (-1, -1)
        batch_size, max_length = inputs.shape[:2]
        return inputs.reshape((batch_size * max_length, *inputs.shape[2:])), (batch_size, max_length)
    distributed = [(key, sequence_distribute(value)) for key, value in inputs.items() if key not in ignore]
    arrays = {key: value[0] for key, value in distributed}
    shape = next(s for _, (_, s) in distributed if s != (-1, -1))
    return cast(MappingT, arrays), shape


@overload
def sequence_undistribute(
    inputs: jax.Array,
    shape: tuple[int, int],
    ignore: Sequence[str] = ...,
) -> jax.Array: ...


@overload
def sequence_undistribute(
    inputs: MappingT,
    shape: tuple[int, int],
    ignore: Sequence[str] = ...,
) -> MappingT: ...


def sequence_undistribute(
    inputs: Union[jax.Array, MappingT],
    shape: tuple[int, int],
    ignore: Sequence[str] = (),
) -> Union[jax.Array, MappingT]:
    if isinstance(inputs, jax.Array):
        return inputs.reshape((shape[0], shape[1], *inputs.shape[1:]))
    return cast(
        MappingT,
        {key: sequence_undistribute(value, shape) for key, value in inputs.items() if key not in ignore},
    )
