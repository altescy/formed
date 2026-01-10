import abc
from typing import Generic, NamedTuple

import jax
from colt import Registrable
from flax import nnx
from typing_extensions import TypeVar

from ..random import require_rngs

_ParamsT = TypeVar("_ParamsT", default=None)


class BaseLabelSampler(nnx.Module, Registrable, Generic[_ParamsT], abc.ABC):
    """Abstract base class for label samplers.

    A LabelSampler defines a strategy for sampling labels based on model logits.

    Type Parameters:
        _ParamsT: Type of additional parameters used during sampling.

    """

    @abc.abstractmethod
    def __call__(self, logits: jax.Array, params: _ParamsT | None = None) -> jax.Array:
        raise NotImplementedError


@BaseLabelSampler.register("argmax")
class ArgmaxLabelSampler(BaseLabelSampler[None]):
    """Label sampler that selects the label with the highest logit."""

    def __call__(self, logits: jax.Array, params: None = None) -> jax.Array:
        return jax.numpy.argmax(logits, axis=-1)


class MultinomialLabelSamplerParams(NamedTuple):
    """Parameters for the MultinomialLabelSampler.

    This class can be extended in the future to include additional parameters
    for sampling if needed.

    Attributes:
        rngs: Random number generators.
        templerature: Sampling temperature to control randomness.
    """

    rngs: nnx.Rngs | None = None
    templerature: float = 1.0


@BaseLabelSampler.register("multinomial")
class MultinomialLabelSampler(BaseLabelSampler[MultinomialLabelSamplerParams]):
    """Label sampler that samples labels from a multinomial distribution defined by the logits."""

    Params = MultinomialLabelSamplerParams

    def __call__(self, logits: jax.Array, params: MultinomialLabelSamplerParams | None = None) -> jax.Array:
        params = params or MultinomialLabelSamplerParams()
        rngs = params.rngs or require_rngs()
        if params.templerature != 1.0:
            logits = logits / params.templerature
        return jax.random.categorical(rngs.key(), logits, axis=-1)
