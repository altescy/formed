from functools import lru_cache
from typing import cast

import jax
from colt import Registrable
from flax import nnx


class PositionEncoder(Registrable):
    def __call__(self, inputs: jax.Array) -> jax.Array:
        raise NotImplementedError


@PositionEncoder.register("sinusoidal")
class SinusoidalPositionEncoder(PositionEncoder):
    def __init__(self, max_length: int = 512) -> None:
        self.max_length = max_length

    @lru_cache(maxsize=1)
    def _encodings(self, features: int) -> jax.Array:
        p, i = jax.numpy.meshgrid(jax.numpy.arange(float(self.max_length)), jax.numpy.arange(features / 2) * 2)
        theta = (p / 1e4 ** (i / features)).T
        encodings = jax.numpy.stack([jax.numpy.sin(theta), jax.numpy.cos(theta)], axis=-1).reshape(
            (self.max_length, features)
        )
        return cast(jax.Array, encodings[None, ...])

    def __call__(self, inputs: jax.Array) -> jax.Array:
        seq_legth, features = inputs.shape[-2:]
        encodings = self._encodings(features)
        return inputs + encodings[:, :seq_legth, :]


@PositionEncoder.register("learnable")
class LearnablePositionEncoder(PositionEncoder):
    def __init__(
        self,
        features: int,
        *,
        rngs: nnx.Rngs,
        max_length: int = 512,
    ) -> None:
        self.embed = nnx.Embed(max_length, features, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        seq_legth, features = inputs.shape[-2:]
        encodings = self.embed(jax.numpy.arange(seq_legth))
        return inputs + encodings
