from collections.abc import Callable
from typing import Literal, Optional, Union

import jax
from colt import Registrable
from flax import nnx

from formed.integrations.flax.utils import masked_pool

from .position_encoders import PositionEncoder


class Seq2VecEncoder(nnx.Module, Registrable):
    def __call__(
        self,
        inputs: jax.Array,
        *,
        seq_lengths: Optional[jax.Array] = None,
    ) -> jax.Array:
        raise NotImplementedError


@Seq2VecEncoder.register("boe")
@Seq2VecEncoder.register("bag_of_embeddings")
class BagOfEmbeddingsSeq2VecEncoder(Seq2VecEncoder):
    def __init__(
        self,
        pooling: Literal["mean", "max", "min", "sum", "hier", "first", "last"] = "mean",
        normalize: bool = False,
        window_size: Optional[int] = None,
    ) -> None:
        self._pooling = pooling
        self._normalize = normalize
        self._window_size = window_size

    def __call__(
        self,
        inputs: jax.Array,
        *,
        seq_lengths: Optional[jax.Array] = None,
    ) -> jax.Array:
        mask: Optional[jax.Array] = None
        if seq_lengths is not None:
            mask = jax.numpy.arange(inputs.shape[1])[None, :] < seq_lengths[:, None]
        return masked_pool(
            inputs,
            mask=mask,
            pooling=self._pooling,
            normalize=self._normalize,
            window_size=self._window_size,
        )
