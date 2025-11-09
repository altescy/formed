import jax
import numpy


def ensure_jax_array(x: numpy.ndarray) -> jax.Array:
    if isinstance(x, jax.Array):
        return x
    return jax.numpy.asarray(x)
